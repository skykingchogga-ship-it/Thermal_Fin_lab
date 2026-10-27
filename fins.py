"""Fin heat-transfer physics: geometry, analytical solutions, finite-volume solver.

Governing 1D fin equation (theta = T - T_inf, s = distance from base):

    d/ds ( k A(s) dtheta/ds ) - h P(s) theta = 0

A(s) is the cross-section area, P(s) the exposed surface per unit length.
For the annular fin, s = r - r1 and A = 2*pi*r*t, P = 4*pi*r (two faces).
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.linalg import solve_banded
from scipy.special import ive, kve

# k in W/m.K, rho in kg/m^3
MATERIALS = {
    "Copper": {"k": 401.0, "rho": 8960.0},
    "Aluminum (pure)": {"k": 237.0, "rho": 2700.0},
    "Aluminum 6061": {"k": 167.0, "rho": 2700.0},
    "Brass": {"k": 109.0, "rho": 8500.0},
    "Carbon steel": {"k": 50.0, "rho": 7850.0},
    "Stainless 304": {"k": 16.2, "rho": 8000.0},
}

SHAPES = ["Rectangular", "Triangular", "Parabolic", "Pin", "Annular"]


@dataclass(frozen=True)
class Fin:
    shape: str
    L: float  # fin length (m); for annular: r2 - r1
    t: float  # base thickness (m); for pin: diameter
    w: float = 0.05  # width (m), straight fins only
    r1: float = 0.02  # base radius (m), annular only


def geometry(fin: Fin, n: int = 401) -> dict:
    """Cross-section A(s), exposed perimeter P(s), thickness profile, tip/base areas."""
    s = np.linspace(0.0, fin.L, n)
    L, t, w = fin.L, fin.t, fin.w
    sh = fin.shape
    if sh == "Rectangular":
        th = np.full(n, t)
        A = w * th
        P = np.full(n, 2 * (w + t))
        A_tip = w * t
    elif sh == "Triangular":
        th = t * (1 - s / L)
        A = w * th
        P = np.full(n, 2 * w)
        A_tip = 0.0
    elif sh == "Parabolic":  # concave parabolic: thickness ~ distance-from-tip squared
        th = t * (1 - s / L) ** 2
        A = w * th
        P = np.full(n, 2 * w)
        A_tip = 0.0
    elif sh == "Pin":
        th = np.full(n, t)
        A = np.full(n, np.pi * t**2 / 4)
        P = np.full(n, np.pi * t)
        A_tip = np.pi * t**2 / 4
    elif sh == "Annular":
        r = fin.r1 + s
        th = np.full(n, t)
        A = 2 * np.pi * r * t
        P = 4 * np.pi * r
        A_tip = 2 * np.pi * (fin.r1 + L) * t
    else:
        raise ValueError(f"Unknown shape: {sh}")
    A_base = float(A[0])
    return dict(s=s, th=th, A=A, P=P, A_tip=float(A_tip), A_base=A_base)


def _csh(x, mL):
    """cosh(x)*exp(-mL), sinh(x)*exp(-mL): overflow-safe for x <= mL."""
    return 0.5 * (np.exp(x - mL) + np.exp(-x - mL)), 0.5 * (np.exp(x - mL) - np.exp(-x - mL))


def analytic(fin: Fin, h, k, Tb, Tinf, s, tip_convective=True):
    """Closed-form temperature profile T(s) [C] and base heat flow q [W]."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    th_b = Tb - Tinf
    ht = h if tip_convective else 0.0
    L, t, w, sh = fin.L, fin.t, fin.w, fin.shape

    if sh in ("Rectangular", "Pin"):
        if sh == "Rectangular":
            A, P = w * t, 2 * (w + t)
        else:
            A, P = np.pi * t**2 / 4, np.pi * t
        m = np.sqrt(h * P / (k * A))
        beta = ht / (m * k)
        mL = m * L
        c_s, s_s = _csh(m * (L - s), mL)
        c_L, s_L = _csh(mL, mL)
        theta = th_b * (c_s + beta * s_s) / (c_L + beta * s_L)
        q = np.sqrt(h * P * k * A) * th_b * (s_L + beta * c_L) / (c_L + beta * s_L)

    elif sh == "Triangular":  # modified Bessel functions I0, I1
        m = np.sqrt(2 * h / (k * t))
        Z = 2 * m * L
        z = 2 * m * np.sqrt(L * np.maximum(L - s, 0.0))
        theta = th_b * ive(0, z) / ive(0, Z) * np.exp(z - Z)
        q = k * w * t * m * th_b * ive(1, Z) / ive(0, Z)

    elif sh == "Parabolic":  # power-law solution theta ~ x^r, x measured from tip
        c = 2 * h * L**2 / (k * t)
        r = (-1 + np.sqrt(1 + 4 * c)) / 2
        theta = th_b * (np.maximum(L - s, 0.0) / L) ** r
        q = k * w * t * th_b * r / L

    elif sh == "Annular":  # I0/K0 combination, solved from base + tip BCs
        m = np.sqrt(2 * h / (k * t))
        r1 = fin.r1
        r2 = r1 + L

        def f1(rr):
            return ive(0, m * rr) * np.exp(m * (rr - r2))

        def d1(rr):
            return m * ive(1, m * rr) * np.exp(m * (rr - r2))

        def f2(rr):
            return kve(0, m * rr) * np.exp(-m * (rr - r1))

        def d2(rr):
            return -m * kve(1, m * rr) * np.exp(-m * (rr - r1))

        M = np.array([[f1(r1), f2(r1)],
                      [k * d1(r2) + ht * f1(r2), k * d2(r2) + ht * f2(r2)]])
        C = np.linalg.solve(M, [th_b, 0.0])
        theta = C[0] * f1(r1 + s) + C[1] * f2(r1 + s)
        q = -k * 2 * np.pi * r1 * t * (C[0] * d1(r1) + C[1] * d2(r1))
    else:
        raise ValueError(sh)

    return Tinf + theta, float(q)


def numeric(fin: Fin, h, k, Tb, Tinf, n=161, tip_convective=True):
    """1D finite-volume solution on n uniform nodes. Returns (T [C], q [W])."""
    g = geometry(fin, n)
    s, P = g["s"], g["P"]
    ds = s[1] - s[0]
    ht = h if tip_convective else 0.0
    th_b = Tb - Tinf

    # Face areas are evaluated exactly at cell midpoints (matters for tapered fins
    # whose area goes to zero at the tip).
    A_face = geometry(fin, 2 * n - 1)["A"][1::2]
    c = k * A_face / ds  # face conductances, length n-1
    m = n - 1  # unknowns: nodes 1..n-1
    cl = c  # coupling to the left neighbour of node i (i = 1..n-1)
    cr = np.append(c[1:], 0.0)  # coupling to the right neighbour
    wgt = np.ones(m)
    wgt[-1] = 0.5  # half control volume at the tip
    d = cl + cr + h * P[1:] * ds * wgt
    d[-1] += ht * g["A_tip"]

    ab = np.zeros((3, m))
    ab[1] = d
    ab[0, 1:] = -cr[:-1]
    ab[2, :-1] = -cl[1:]
    rhs = np.zeros(m)
    rhs[0] = cl[0] * th_b
    theta = np.concatenate([[th_b], solve_banded((1, 1), ab, rhs)])

    q = h * trapezoid(P * theta, s) + ht * g["A_tip"] * theta[-1]
    return Tinf + theta, float(q)


def performance(fin: Fin, h, k, rho, Tb, Tinf, q, n_geo=201) -> dict:
    """Efficiency, effectiveness, mass and Biot number for a given base heat flow."""
    g = geometry(fin, n_geo)
    th_b = Tb - Tinf
    A_surf = trapezoid(g["P"], g["s"]) + g["A_tip"]
    vol = trapezoid(g["A"], g["s"])
    return dict(
        eta=q / (h * A_surf * th_b),
        eps=q / (h * g["A_base"] * th_b),
        mass=rho * vol,
        biot=h * (fin.t / 2) / k,
        A_surf=A_surf,
    )


def sweep(shapes, materials, L_range, t_range, h, Tb, Tinf, w, r1, tip_convective=True, n=14):
    """Analytical design sweep over shape x material x length x thickness."""
    import pandas as pd

    rows = []
    s2 = None
    for shape in shapes:
        for L in np.linspace(L_range[0], L_range[1], n):
            for t in np.linspace(t_range[0], t_range[1], n):
                fin = Fin(shape, float(L), float(t), w, r1)
                s2 = np.array([0.0, L])
                for mat in materials:
                    k, rho = MATERIALS[mat]["k"], MATERIALS[mat]["rho"]
                    _, q = analytic(fin, h, k, Tb, Tinf, s2, tip_convective)
                    pf = performance(fin, h, k, rho, Tb, Tinf, q, n_geo=81)
                    rows.append({
                        "Shape": shape, "Material": mat,
                        "L (mm)": L * 1e3, "t (mm)": t * 1e3,
                        "q (W)": q, "η (%)": pf["eta"] * 100,
                        "Mass (g)": pf["mass"] * 1e3,
                        "q/mass (W/kg)": q / pf["mass"],
                        "Biot": pf["biot"],
                    })
    return pd.DataFrame(rows)
