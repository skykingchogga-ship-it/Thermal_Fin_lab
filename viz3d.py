"""3D fin models coloured by the temperature solution (Plotly)."""
import numpy as np
import plotly.graph_objects as go

import fins as F

GREY = "#8d939c"
GREY_SCALE = [[0, GREY], [1, GREY]]
LIGHT = dict(ambient=0.55, diffuse=0.85, specular=0.25, roughness=0.55, fresnel=0.1)
MM = 1e3


class _Scene:
    def __init__(self, cmin, cmax, unit="°C", fmt=".1f"):
        self.fig = go.Figure()
        self.cmin, self.cmax = cmin, cmax
        self.unit, self.fmt = unit, fmt
        self.first = True

    def temp(self, X, Y, Z, C):
        """Surface coloured by the field C (same shape as X, Y, Z)."""
        kw = dict(colorbar=dict(title=self.unit, len=0.7)) if self.first else {}
        self.fig.add_trace(go.Surface(
            x=X * MM, y=Y * MM, z=Z * MM, surfacecolor=C, colorscale="Inferno",
            cmin=self.cmin, cmax=self.cmax, showscale=self.first, lighting=LIGHT,
            hovertemplate="%{surfacecolor:" + self.fmt + "} " + self.unit + "<extra></extra>", **kw))
        self.first = False

    def grey(self, X, Y, Z):
        self.fig.add_trace(go.Surface(
            x=X * MM, y=Y * MM, z=Z * MM, surfacecolor=np.zeros_like(X), colorscale=GREY_SCALE,
            showscale=False, lighting=LIGHT, hoverinfo="skip"))

    def box(self, x0, x1, y0, y1, z0, z1):
        self.fig.add_trace(go.Mesh3d(
            x=np.array([x0, x0, x1, x1, x0, x0, x1, x1]) * MM,
            y=np.array([y0, y1, y1, y0, y0, y1, y1, y0]) * MM,
            z=np.array([z0, z0, z0, z0, z1, z1, z1, z1]) * MM,
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=GREY, flatshading=True, lighting=LIGHT, hoverinfo="skip"))


def _straight(sc, fin, s, T, ex, wall):
    ns, nz, nv = 140, 14, 6
    g = F.geometry(fin, ns)
    ss, th = g["s"], g["th"] * ex
    Tt = np.interp(ss, s, T)
    zs = np.linspace(-fin.w / 2, fin.w / 2, nz)
    v = np.linspace(-1, 1, nv)

    X = np.tile(ss, (nz, 1))
    Z = np.tile(zs[:, None], (1, ns))
    C = np.tile(Tt, (nz, 1))
    Yt = np.tile(th / 2, (nz, 1))
    sc.temp(X, Yt, Z, C)  # top face
    sc.temp(X, -Yt, Z, C)  # bottom face

    Xs = np.tile(ss, (nv, 1))
    Ys = v[:, None] * (th / 2)[None, :]
    Cs = np.tile(Tt, (nv, 1))
    for zc in (-fin.w / 2, fin.w / 2):  # side faces
        sc.temp(Xs, Ys, np.full_like(Xs, zc), Cs)

    for x_end, th_e, T_e in ((0.0, th[0], Tt[0]), (fin.L, th[-1], Tt[-1])):  # end caps
        if th_e > 1e-9:
            Xc = np.full((nv, nz), x_end)
            Yc = (v[:, None] * th_e / 2) * np.ones((1, nz))
            Zc = np.tile(zs, (nv, 1))
            sc.temp(Xc, Yc, Zc, np.full((nv, nz), T_e))

    if wall:
        wt, H, hw = 0.15 * fin.L, max(3 * th[0], 0.3 * fin.L), 0.6 * fin.w
        sc.box(-wt, 0.0, -H / 2, H / 2, -hw, hw)
    return dict(eye=dict(x=1.4, y=1.4, z=1.0))


def _pin(sc, fin, s, T, ex, wall):
    ns, nphi, nrho = 120, 48, 8
    ss = np.linspace(0, fin.L, ns)
    Tt = np.interp(ss, s, T)
    r = fin.t / 2 * ex
    phi = np.linspace(0, 2 * np.pi, nphi)
    X = np.tile(ss, (nphi, 1))
    Y = r * np.cos(phi)[:, None] * np.ones((1, ns))
    Z = r * np.sin(phi)[:, None] * np.ones((1, ns))
    sc.temp(X, Y, Z, np.tile(Tt, (nphi, 1)))

    rho = np.linspace(0, r, nrho)
    for x_end, T_e in ((fin.L, Tt[-1]), (0.0, Tt[0])):
        Xc = np.full((nrho, nphi), x_end)
        Yc = rho[:, None] * np.cos(phi)[None, :]
        Zc = rho[:, None] * np.sin(phi)[None, :]
        sc.temp(Xc, Yc, Zc, np.full((nrho, nphi), T_e))

    if wall:
        wt, hs = 0.15 * fin.L, max(3 * r, 0.15 * fin.L)
        sc.box(-wt, 0.0, -hs, hs, -hs, hs)
    return dict(eye=dict(x=1.3, y=1.3, z=0.9))


def _annular(sc, fin, s, T, ex, wall):
    nr, nphi = 70, 96
    r1, r2, tt = fin.r1, fin.r1 + fin.L, fin.t / 2 * ex
    rr = np.linspace(r1, r2, nr)
    Tr = np.interp(rr - r1, s, T)
    phi = np.linspace(0, 2 * np.pi, nphi)
    Xr = rr[:, None] * np.cos(phi)[None, :]
    Yr = rr[:, None] * np.sin(phi)[None, :]
    Cr = np.tile(Tr[:, None], (1, nphi))
    sc.temp(Xr, Yr, np.full_like(Xr, tt), Cr)  # top face
    sc.temp(Xr, Yr, np.full_like(Xr, -tt), Cr)  # bottom face

    zz = np.linspace(-tt, tt, 4)
    for rim_r, T_e in ((r2, Tr[-1]), (r1, Tr[0])):  # outer and inner rims
        Xc = rim_r * np.cos(phi)[None, :] * np.ones((4, 1))
        Yc = rim_r * np.sin(phi)[None, :] * np.ones((4, 1))
        Zc = zz[:, None] * np.ones((1, nphi))
        sc.temp(Xc, Yc, Zc, np.full((4, nphi), T_e))

    if wall:  # central tube the fin is mounted on
        hz = max(1.5 * tt, 0.25 * fin.L)
        zc = np.linspace(-hz, hz, 4)
        sc.grey(r1 * np.cos(phi)[None, :] * np.ones((4, 1)),
                r1 * np.sin(phi)[None, :] * np.ones((4, 1)),
                zc[:, None] * np.ones((1, nphi)))
        rho = np.linspace(0, r1, 6)
        for zend in (-hz, hz):
            sc.grey(rho[:, None] * np.cos(phi)[None, :],
                    rho[:, None] * np.sin(phi)[None, :],
                    np.full((6, nphi), zend))
    return dict(eye=dict(x=1.1, y=-1.5, z=1.1))


def fin_3d_figure(fin: F.Fin, s, T, Tinf, Tb, ex=1.0, wall=True) -> go.Figure:
    """Rotatable 3D model of the fin, surfaces coloured by T(s) in °C.

    s, T: distance from base (m) and temperature (°C); ex: thickness exaggeration factor.
    """
    sc = _Scene(Tinf, Tb)
    builder = {"Rectangular": _straight, "Triangular": _straight, "Parabolic": _straight,
               "Pin": _pin, "Annular": _annular}[fin.shape]
    eye = builder(sc, fin, s, T, ex, wall)
    return finish(sc.fig, f"{fin.shape} fin — 3D temperature model", eye)


def finish(fig, title, camera):
    """Common layout for the 3D figures."""
    axis = dict(showbackground=False, gridcolor="rgba(150,150,150,0.25)", zeroline=False)
    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=40, b=0), title=title,
        scene=dict(aspectmode="data", camera=camera,
                   xaxis=dict(title="x (mm)", **axis), yaxis=dict(title="y (mm)", **axis),
                   zaxis=dict(title="z (mm)", **axis)))
    return fig
