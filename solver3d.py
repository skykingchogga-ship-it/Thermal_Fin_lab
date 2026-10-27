"""Steady 3D heat conduction in fins (finite volume + sparse direct solve).

* Straight fins (rectangular, triangular, parabolic): Cartesian grid on one quarter of the
  fin (symmetry planes y = 0 and z = 0 are adiabatic). Tapered surfaces use cut cells, so the
  outline is followed accurately.
* Pin and annular fins: exact axisymmetric (r, z) finite-volume solution, revolved for display.

All faces exposed to air lose heat by convection with coefficient h; the base is isothermal.
Conduction is solved in the whole volume, so temperature varies across the thickness too.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

EPS = 1e-6

RES = {
    "Coarse": dict(nx=40, ny=6, nz=10, pin=(8, 50), ann=(40, 4)),
    "Medium": dict(nx=60, ny=8, nz=14, pin=(12, 90), ann=(70, 6)),
    "Fine": dict(nx=90, ny=12, nz=20, pin=(16, 140), ann=(110, 8)),
}


def half_thickness(fin, x):
    """Half thickness of a straight fin at distance x from the base (m)."""
    x = np.asarray(x, dtype=float)
    t2 = fin.t / 2
    if fin.shape == "Rectangular":
        return np.full_like(x, t2)
    if fin.shape == "Triangular":
        return t2 * (1 - x / fin.L)
    if fin.shape == "Parabolic":
        return t2 * (1 - x / fin.L) ** 2
    raise ValueError(f"{fin.shape} is not a straight fin")


class _System:
    """Sparse conductance network: sum_nb G (T_nb - T) + sum_bc G (T_bc - T) = 0."""

    def __init__(self, n):
        self.n = n
        self.diag = np.zeros(n)
        self.rhs = np.zeros(n)
        self.r, self.c, self.v = [], [], []
        self.base, self.conv = [], []

    def pair(self, a, b, G):
        a, b, G = np.ravel(a), np.ravel(b), np.ravel(G)
        self.diag += np.bincount(a, G, self.n) + np.bincount(b, G, self.n)
        self.r += [a, b]
        self.c += [b, a]
        self.v += [-G, -G]

    def fixed(self, a, G, theta, kind):
        a, G = np.ravel(a), np.ravel(G)
        self.diag += np.bincount(a, G, self.n)
        if theta:
            self.rhs += np.bincount(a, G * theta, self.n)
        (self.base if kind == "base" else self.conv).append((a, G))

    def solve(self, inactive=None, theta_b=0.0):
        if inactive is not None:
            self.diag[inactive] = 1.0
            self.rhs[inactive] = 0.0
        A = coo_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.c))),
                       shape=(self.n, self.n)).tocsc() + diags(self.diag)
        theta = spsolve(A.tocsc(), self.rhs)
        q_base = sum(float(np.sum(G * (theta_b - theta[a]))) for a, G in self.base)
        q_conv = sum(float(np.sum(G * theta[a])) for a, G in self.conv)
        return theta, q_base, q_conv


def _conv(S, h, k, cells, A, dist):
    """Convective boundary through a conduction path of length `dist` (cell centre -> surface)."""
    A = np.ravel(A).astype(float)
    ok = A > 0
    dist = np.broadcast_to(np.ravel(dist), A.shape)
    G = A[ok] / (1.0 / h + dist[ok] / k)
    S.fixed(np.ravel(cells)[ok], G, 0.0, "conv")


def _solve_straight(fin, h, k, Tb, Tinf, tip, nx, ny, nz):
    L, w, t = fin.L, fin.w, fin.t
    dx, dy, dz = L / nx, (t / 2) / ny, (w / 2) / nz
    xc, yc, zc = (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, (np.arange(nz) + 0.5) * dz
    half = half_thickness(fin, xc)
    hy = np.clip(half[:, None] - np.arange(ny)[None, :] * dy, 0.0, dy)  # cut-cell heights
    hy[hy < EPS * dy] = 0.0
    HY = np.repeat(hy[:, :, None], nz, axis=2)
    act = HY > 0
    N = nx * ny * nz
    idx = np.arange(N).reshape(nx, ny, nz)
    S = _System(N)
    thb = Tb - Tinf

    # conduction between active neighbours
    m = act[:-1] & act[1:]
    S.pair(idx[:-1][m], idx[1:][m], (k * dz * np.minimum(HY[:-1], HY[1:]) / dx)[m])
    m = act[:, :-1] & act[:, 1:]
    den = np.where(m, 0.5 * (HY[:, :-1] + HY[:, 1:]), 1.0)
    S.pair(idx[:, :-1][m], idx[:, 1:][m], (k * dx * dz / den)[m])
    m = act[:, :, :-1]
    S.pair(idx[:, :, :-1][m], idx[:, :, 1:][m], (k * dx * HY[:, :, :-1] / dz)[m])

    # convection: top surface of each column
    nxt = np.zeros_like(act)
    nxt[:, :-1, :] = act[:, 1:, :]
    top = act & ~nxt
    slope = np.gradient(half, dx) if nx > 1 else np.zeros(1)
    stretch = np.sqrt(1.0 + slope**2)[:, None, None] * np.ones((nx, ny, nz))  # sloped surface area
    _conv(S, h, k, idx[top], dx * dz * stretch[top], HY[top] / 2)
    # side face z = w/2
    sm = act[:, :, -1]
    _conv(S, h, k, idx[:, :, -1][sm], dx * HY[:, :, -1][sm], dz / 2)
    # tip face x = L (tapered surfaces are covered by the stretched top-face area above)
    if tip:
        e = HY[-1]
        ok = e > EPS * dy
        _conv(S, h, k, idx[-1][ok], dz * e[ok], dx / 2)
    # isothermal base at x = 0
    a0 = act[0]
    A0 = dz * HY[0][a0]
    S.fixed(idx[0][a0], k * A0 / (dx / 2), thb, "base")

    theta, qb, qc = S.solve(inactive=~act.ravel(), theta_b=thb)
    th3 = theta.reshape(nx, ny, nz)
    jtop = np.maximum(act[:, :, 0].sum(axis=1) - 1, 0)
    jj = np.minimum(np.arange(ny)[None, :], jtop[:, None])
    Tf = Tinf + th3[np.arange(nx)[:, None], jj]  # inactive cells filled from the surface cell
    T = np.where(act, Tinf + th3, np.nan)
    tmin = np.nanmin(T, axis=(1, 2), keepdims=True)
    dev = T - tmin
    devf = dev.copy()
    devf = np.nan_to_num(devf, nan=0.0)
    devf = devf[np.arange(nx)[:, None], jj]
    w_tip = HY[-1]
    T_tip = float(np.nansum(np.where(act[-1], T[-1], 0.0) * w_tip) / w_tip.sum())
    return dict(kind="straight", T=T, Tf=Tf, dev=dev, devf=devf, x=xc, y=yc, z=zc, jtop=jtop,
                q=4 * qb, q_conv=4 * qc, dT_cross=float(np.nanmax(dev)), T_tip=T_tip)


def _axisym(r_edges, z_edges, k, h, bc, Tb, Tinf):
    """Axisymmetric finite volume. bc: sides rmin/rmax/zmin/zmax -> 'base' | 'conv' | 'adia'."""
    nr, nz = len(r_edges) - 1, len(z_edges) - 1
    dr, dz = r_edges[1] - r_edges[0], z_edges[1] - z_edges[0]
    idx = np.arange(nr * nz).reshape(nr, nz)
    S = _System(nr * nz)
    thb = Tb - Tinf
    Aax = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)  # axial face area per radial cell

    Gr = k * 2 * np.pi * r_edges[1:-1] * dz / dr
    S.pair(idx[:-1], idx[1:], np.broadcast_to(Gr[:, None], (nr - 1, nz)))
    S.pair(idx[:, :-1], idx[:, 1:], np.broadcast_to((k * Aax / dz)[:, None], (nr, nz - 1)))

    def boundary(cells, A, dist, kind):
        if kind == "adia":
            return
        if kind == "base":
            A = np.ravel(A).astype(float)
            ok = A > 0
            S.fixed(np.ravel(cells)[ok], k * A[ok] / dist, thb, "base")
        else:
            _conv(S, h, k, cells, A, dist)

    boundary(idx[0, :], np.full(nz, 2 * np.pi * r_edges[0] * dz), dr / 2, bc["rmin"])
    boundary(idx[-1, :], np.full(nz, 2 * np.pi * r_edges[-1] * dz), dr / 2, bc["rmax"])
    boundary(idx[:, 0], Aax, dz / 2, bc["zmin"])
    boundary(idx[:, -1], Aax, dz / 2, bc["zmax"])
    theta, qb, qc = S.solve(theta_b=thb)
    return Tinf + theta.reshape(nr, nz), qb, qc, Aax


def _solve_pin(fin, h, k, Tb, Tinf, tip, nr, nx):
    R = fin.t / 2
    re, ze = np.linspace(0, R, nr + 1), np.linspace(0, fin.L, nx + 1)
    bc = dict(rmin="adia", rmax="conv", zmin="base", zmax="conv" if tip else "adia")
    T, qb, qc, Aax = _axisym(re, ze, k, h, bc, Tb, Tinf)
    rc, xc = 0.5 * (re[:-1] + re[1:]), 0.5 * (ze[:-1] + ze[1:])
    dev = T - T.min(axis=0, keepdims=True)
    return dict(kind="pin", T=T, Tf=T, dev=dev, devf=dev, r=rc, x=xc, q=qb, q_conv=qc,
                dT_cross=float(dev.max()), T_tip=float(np.sum(T[:, -1] * Aax) / Aax.sum()))


def _solve_annular(fin, h, k, Tb, Tinf, tip, nr, nz):
    r1, r2 = fin.r1, fin.r1 + fin.L
    re, ze = np.linspace(r1, r2, nr + 1), np.linspace(0, fin.t / 2, nz + 1)
    bc = dict(rmin="base", rmax="conv" if tip else "adia", zmin="adia", zmax="conv")
    T, qb, qc, _ = _axisym(re, ze, k, h, bc, Tb, Tinf)
    rc, zc = 0.5 * (re[:-1] + re[1:]), 0.5 * (ze[:-1] + ze[1:])
    dev = T - T.min(axis=1, keepdims=True)
    return dict(kind="annular", T=T, Tf=T, dev=dev, devf=dev, r=rc, z=zc, q=2 * qb, q_conv=2 * qc,  # modelled half thickness: z = 0 is a symmetry plane
                dT_cross=float(dev.max()), T_tip=float(T[-1].mean()))


def solve_3d(fin, h, k, Tb, Tinf, tip=True, resolution="Medium"):
    """Full 3D conduction solution. Returns a dict with the field and summary numbers.

    q is the total heat flow from the base into the fin (W); for annular fins the axisymmetric
    solution already covers the full ring (factor 2*pi built in via the face areas).
    """
    res = RES[resolution]
    if fin.shape in ("Rectangular", "Triangular", "Parabolic"):
        return _solve_straight(fin, h, k, Tb, Tinf, tip, res["nx"], res["ny"], res["nz"])
    if fin.shape == "Pin":
        return _solve_pin(fin, h, k, Tb, Tinf, tip, *res["pin"])
    if fin.shape == "Annular":
        return _solve_annular(fin, h, k, Tb, Tinf, tip, *res["ann"])
    raise ValueError(fin.shape)
