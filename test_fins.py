"""Validation tests: analytical solutions vs finite-volume solver, plus physical sanity checks."""
import numpy as np
import pytest

import fins as F

K, H, TB, TINF = 167.0, 40.0, 120.0, 25.0


def make(shape):
    return F.Fin(shape, 0.06, 0.003, 0.05, 0.02)


@pytest.mark.parametrize("shape", F.SHAPES)
@pytest.mark.parametrize("wall", [True, False])
def test_3d_model_builds_with_valid_colours(shape, wall):
    import viz3d

    fin = make(shape)
    s = F.geometry(fin, 101)["s"]
    T, _ = F.analytic(fin, H, K, TB, TINF, s)
    fig = viz3d.fin_3d_figure(fin, s, T, TINF, TB, ex=3.0, wall=wall)
    assert len(fig.data) >= 3
    for tr in fig.data:
        if tr.type == "surface" and tr.showscale is not None and tr.colorscale[0][1] != viz3d.GREY:
            c = np.asarray(tr.surfacecolor)
            assert np.all(np.isfinite(c))
            assert c.min() >= TINF - 1e-6 and c.max() <= TB + 1e-6


@pytest.mark.parametrize("shape", F.SHAPES)
@pytest.mark.parametrize("tip", [True, False])
def test_numeric_matches_analytic(shape, tip):
    fin = make(shape)
    s = F.geometry(fin, 801)["s"]
    T_a, q_a = F.analytic(fin, H, K, TB, TINF, s, tip)
    T_n, q_n = F.numeric(fin, H, K, TB, TINF, 801, tip)
    assert abs(q_n - q_a) / q_a < 0.01
    err = np.abs(T_n - T_a)
    if shape == "Parabolic":
        # theta ~ x^0.4 has an infinite slope at the zero-area tip, so the last cell is
        # inherently coarse; check everywhere else.
        err = err[: int(0.99 * len(err))]
    assert np.max(err) < 0.5


@pytest.mark.parametrize("shape", F.SHAPES)
def test_efficiency_between_zero_and_one(shape):
    fin = make(shape)
    _, q = F.analytic(fin, H, K, TB, TINF, np.array([0.0, fin.L]))
    pf = F.performance(fin, H, K, 2700.0, TB, TINF, q)
    assert 0.0 < pf["eta"] <= 1.0
    assert pf["eps"] > 1.0  # a useful fin must beat the bare base area


@pytest.mark.parametrize("shape", F.SHAPES)
def test_infinite_conductivity_gives_unit_efficiency(shape):
    fin = make(shape)
    _, q = F.analytic(fin, H, 1e7, TB, TINF, np.array([0.0, fin.L]))
    pf = F.performance(fin, H, 1e7, 2700.0, TB, TINF, q)
    assert pf["eta"] > 0.999


def test_long_rectangular_fin_limit():
    fin = F.Fin("Rectangular", 5.0, 0.003, 0.05)
    _, q = F.analytic(fin, H, K, TB, TINF, np.array([0.0, fin.L]))
    A, P = fin.w * fin.t, 2 * (fin.w + fin.t)
    assert q == pytest.approx(np.sqrt(H * P * K * A) * (TB - TINF), rel=1e-6)


@pytest.mark.parametrize("shape", F.SHAPES)
def test_temperature_decreases_along_fin(shape):
    fin = make(shape)
    s = F.geometry(fin, 101)["s"]
    T, _ = F.analytic(fin, H, K, TB, TINF, s)
    assert T[0] == pytest.approx(TB)
    assert np.all(np.diff(T) <= 1e-9)
    assert T[-1] >= TINF  # concave parabolic tip is exactly ambient in the ideal solution
