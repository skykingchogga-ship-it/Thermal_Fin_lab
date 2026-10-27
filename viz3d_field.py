"""3D fin models coloured by the solved 3D temperature field, with an optional cutaway."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import solver3d as S
from viz3d import _Scene, finish


def _interp2(xc, yc, field, X, Y):
    """Bilinear interpolation of field[xc, yc] at (X, Y); same shape as X."""
    f = RegularGridInterpolator((xc, yc), field, bounds_error=False, fill_value=None)
    out = f(np.stack([np.ravel(X), np.ravel(Y)], axis=-1)).reshape(np.shape(X))
    return np.clip(out, np.nanmin(field), np.nanmax(field))


# --------------------------------------------------------------------------- straight fins
def _straight(sc, fin, res, F3, ex, wall, cut):
    L, w = fin.L, fin.w
    xc, yc, zc, jt = res["x"], res["y"], res["z"], res["jtop"]
    nx = len(xc)
    nxp, nzp, nvp = 121, 15, 7
    xp, zp, vp = np.linspace(0, L, nxp), np.linspace(0, w / 2, nzp), np.linspace(0, 1, nvp)
    half = S.half_thickness(fin, xp)
    hd = half * ex

    XZx, XZz = np.meshgrid(xp, zp)  # (nzp, nxp)
    C_top = _interp2(xc, zc, F3[np.arange(nx), jt, :], XZx, XZz)
    XVx = np.tile(xp, (nvp, 1))  # (nvp, nxp)
    YQ = vp[:, None] * half[None, :]
    YD = vp[:, None] * hd[None, :]
    C_side = _interp2(xc, yc, F3[:, :, -1], XVx, YQ)

    def cap(index, x_end, half_e, sy, sz):
        if half_e <= 1e-9:
            return
        Yq = np.tile(vp[:, None] * half_e, (1, nzp))
        Zq = np.tile(zp, (nvp, 1))
        C = _interp2(yc, zc, F3[index], Yq, Zq)
        sc.temp(np.full((nvp, nzp), x_end), sy * Yq * ex, sz * Zq, C)

    for sy in (1, -1):
        for sz in (1, -1):
            if cut and sy == 1 and sz == 1:
                continue  # removed quadrant
            sc.temp(XZx, sy * np.tile(hd, (nzp, 1)), sz * XZz, C_top)  # top / bottom skin
            sc.temp(XVx, sy * YD, np.full_like(XVx, sz * w / 2), C_side)  # side skin
            cap(-1, L, half[-1], sy, sz)
            if not wall:
                cap(0, 0.0, half[0], sy, sz)

    if cut:  # internal faces exposed by the cut
        sc.temp(XVx, YD, np.zeros_like(XVx), _interp2(xc, yc, F3[:, :, 0], XVx, YQ))
        sc.temp(XZx, np.zeros_like(XZx), XZz, _interp2(xc, zc, F3[:, 0, :], XZx, XZz))
    if wall:
        wt, H, hw = 0.15 * L, max(3 * hd[0], 0.3 * L), 0.6 * w
        sc.box(-wt, 0.0, -H / 2, H / 2, -hw, hw)
    return dict(eye=dict(x=1.0, y=1.4, z=1.4)) if cut else dict(eye=dict(x=1.4, y=1.4, z=1.0))


# --------------------------------------------------------------------------- pin fin
def _pin(sc, fin, res, F3, ex, wall, cut):
    L, R = fin.L, fin.t / 2
    rc, xc = res["r"], res["x"]
    nxp, nphi, nvp = 121, 97, 8
    phi1 = 1.5 * np.pi if cut else 2 * np.pi
    phi = np.linspace(0, phi1, nphi)
    xp, vp = np.linspace(0, L, nxp), np.linspace(0, 1, nvp)
    Rd = R * ex

    C = np.interp(xp, xc, F3[-1, :])
    sc.temp(np.tile(xp, (nphi, 1)), Rd * np.cos(phi)[:, None] * np.ones((1, nxp)),
            Rd * np.sin(phi)[:, None] * np.ones((1, nxp)), np.tile(C, (nphi, 1)))
    discs = [(L, F3[:, -1])] + ([] if wall else [(0.0, F3[:, 0])])
    for xe, col in discs:
        Cd = np.interp(vp * R, rc, col)
        sc.temp(np.full((nvp, nphi), xe), vp[:, None] * Rd * np.cos(phi)[None, :],
                vp[:, None] * Rd * np.sin(phi)[None, :], np.tile(Cd[:, None], (1, nphi)))
    if cut:
        for pc in (0.0, phi1):
            Xp = np.tile(xp, (nvp, 1))
            Cp = _interp2(rc, xc, F3, vp[:, None] * R * np.ones((1, nxp)), Xp)
            sc.temp(Xp, vp[:, None] * Rd * np.cos(pc) * np.ones((1, nxp)),
                    vp[:, None] * Rd * np.sin(pc) * np.ones((1, nxp)), Cp)
    if wall:
        wt, hs = 0.15 * L, max(3 * Rd, 0.15 * L)
        sc.box(-wt, 0.0, -hs, hs, -hs, hs)
    return dict(eye=dict(x=1.2, y=1.3, z=-1.3)) if cut else dict(eye=dict(x=1.3, y=1.3, z=0.9))


# --------------------------------------------------------------------------- annular fin
def _annular(sc, fin, res, F3, ex, wall, cut):
    r1, r2 = fin.r1, fin.r1 + fin.L
    hz = fin.t / 2
    hzd = hz * ex
    rc, zc = res["r"], res["z"]
    nrp, nphi, nvp = 80, 97, 7
    phi1 = 1.5 * np.pi if cut else 2 * np.pi
    phi = np.linspace(0, phi1, nphi)
    rp, vp = np.linspace(r1, r2, nrp), np.linspace(-1, 1, nvp)

    Xr, Yr = rp[:, None] * np.cos(phi)[None, :], rp[:, None] * np.sin(phi)[None, :]
    Ct = np.tile(np.interp(rp, rc, F3[:, -1])[:, None], (1, nphi))
    sc.temp(Xr, Yr, np.full_like(Xr, hzd), Ct)  # top face
    sc.temp(Xr, Yr, np.full_like(Xr, -hzd), Ct)  # bottom face

    rims = [(r2, F3[-1, :])] + ([] if wall else [(r1, F3[0, :])])
    for rr, col in rims:
        Cr = np.tile(np.interp(np.abs(vp) * hz, zc, col)[:, None], (1, nphi))
        sc.temp(rr * np.cos(phi)[None, :] * np.ones((nvp, 1)), rr * np.sin(phi)[None, :] * np.ones((nvp, 1)),
                vp[:, None] * hzd * np.ones((1, nphi)), Cr)
    if cut:
        R2, V2 = np.meshgrid(rp, vp)  # (nvp, nrp)
        Cp = _interp2(rc, zc, F3, R2, np.abs(V2) * hz)
        for pc in (0.0, phi1):
            sc.temp(R2 * np.cos(pc), R2 * np.sin(pc), V2 * hzd, Cp)
    if wall:  # central tube the fin is mounted on
        hzw = max(1.5 * hzd, 0.25 * fin.L)
        zz = np.linspace(-hzw, hzw, 4)
        full = np.linspace(0, 2 * np.pi, nphi)
        sc.grey(r1 * np.cos(full)[None, :] * np.ones((4, 1)), r1 * np.sin(full)[None, :] * np.ones((4, 1)),
                zz[:, None] * np.ones((1, nphi)))
        rho = np.linspace(0, r1, 6)
        for ze in (-hzw, hzw):
            sc.grey(rho[:, None] * np.cos(full)[None, :], rho[:, None] * np.sin(full)[None, :],
                    np.full((6, nphi), ze))
    return dict(eye=dict(x=1.2, y=-1.2, z=1.2)) if cut else dict(eye=dict(x=1.1, y=-1.5, z=1.1))


def field_figure(fin, res, Tinf, Tb, mode="T", ex=1.0, wall=True, cut=True):
    """3D model coloured by the solved field.

    mode "T": temperature (°C). mode "dev": temperature minus the coolest point of the same
    cross-section, which makes through-thickness (3D) variation visible even when it is small.
    """
    if mode == "T":
        F3 = res["Tf"]
        sc = _Scene(Tinf, Tb, "°C", ".2f")
        title = f"{fin.shape} fin — 3D conduction solution"
    else:
        F3 = res["devf"]
        sc = _Scene(0.0, max(res["dT_cross"], 1e-12), "K", ".3g")
        title = f"{fin.shape} fin — internal temperature rise above the coolest point of each section"
    builder = {"straight": _straight, "pin": _pin, "annular": _annular}[res["kind"]]
    eye = builder(sc, fin, res, F3, ex, wall, cut)
    return finish(sc.fig, title, eye)
