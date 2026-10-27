"""3D conduction solver: agreement with the 1D theory, energy balance, physical behaviour."""
import numpy as np
import pytest

import fins as F
import solver3d as S
import viz3d_field as V

K, H, TB, TINF = 167.0, 40.0, 120.0, 25.0

# Tolerances on heat flow vs the 1D analytical solution at low Biot number.
# Tapered fins: the 1D thin-fin form (P = 2w) ignores the side faces, so 3D carries a little more.
TOL = {"Rectangular": 0.005, "Pin": 0.005, "Annular": 0.005, "Triangular": 0.04, "Parabolic": 0.04}


def make(shape, w=0.05):
    return F.Fin(shape, 0.06, 0.003, w, 0.02)


@pytest.mark.parametrize("shape", F.SHAPES)
@pytest.mark.parametrize("tip", [True, False])
def test_3d_heat_flow_matches_1d_at_low_biot(shape, tip):
    fin = make(shape)
    _, q1 = F.analytic(fin, H, K, TB, TINF, np.array([0.0, fin.L]), tip)
    r = S.solve_3d(fin, H, K, TB, TINF, tip, "Medium")
    assert abs(r["q"] / q1 - 1) < TOL[shape]


@pytest.mark.parametrize("shape", ["Triangular", "Parabolic"])
def test_wide_tapered_fin_converges_to_1d(shape):
    fin = make(shape, w=0.5)  # edge effects vanish
    _, q1 = F.analytic(fin, H, K, TB, TINF, np.array([0.0, fin.L]), True)
    r = S.solve_3d(fin, H, K, TB, TINF, True, "Medium")
    assert abs(r["q"] / q1 - 1) < 0.01


@pytest.mark.parametrize("shape", F.SHAPES)
@pytest.mark.parametrize("res", ["Coarse", "Medium"])
def test_energy_balance(shape, res):
    r = S.solve_3d(make(shape), H, K, TB, TINF, True, res)
    assert abs(r["q"] - r["q_conv"]) / r["q"] < 1e-8


@pytest.mark.parametrize("shape", F.SHAPES)
def test_field_bounded_and_cools_along_fin(shape):
    r = S.solve_3d(make(shape), H, K, TB, TINF, True, "Medium")
    T = r["T"][np.isfinite(r["T"])]
    assert T.min() >= TINF - 1e-9 and T.max() <= TB + 1e-9
    assert r["T_tip"] < TB


def test_internal_gradient_grows_with_biot():
    fin = F.Fin("Rectangular", 0.05, 0.010, 0.05)
    lo = S.solve_3d(fin, 10.0, 237.0, TB, TINF)["dT_cross"]
    hi = S.solve_3d(fin, 300.0, 16.2, TB, TINF)["dT_cross"]
    assert hi > 100 * lo
    # in the high-Biot case the core is hotter than the surface (through-thickness gradient)
    assert hi > 1.0


@pytest.mark.parametrize("shape", F.SHAPES)
@pytest.mark.parametrize("mode", ["T", "dev"])
@pytest.mark.parametrize("cut", [True, False])
def test_field_figures_build(shape, mode, cut):
    fin = make(shape)
    r = S.solve_3d(fin, H, K, TB, TINF, True, "Coarse")
    fig = V.field_figure(fin, r, TINF, TB, mode=mode, ex=2.0, wall=True, cut=cut)
    assert len(fig.data) >= 3
    for tr in fig.data:
        if tr.type == "surface" and tr.colorscale[0][1] != "#8d939c":
            assert np.all(np.isfinite(np.asarray(tr.surfacecolor)))
