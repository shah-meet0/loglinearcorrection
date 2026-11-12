"""Tests for the reading (b) panel estimator and the n_folds=1 paths."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from sim_readingb import simulate, truth  # noqa: E402

from loglinearcorrection.panel_readingb import PanelReadingB  # noqa: E402


def _model(d, design="none", time_effects=False, **kw):
    fe = ["unit", "t"] if time_effects else ["unit"]
    cols = (["s", "w"] if design != "none" else ["s"]) + fe
    return PanelReadingB(d["y"], d["frame"][cols], fixed_effects=fe, unit="unit", interest="s", **kw)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parametrize("design", ["none", "tinv", "tvar"])
def test_recovers_reading_b_targets(design):
    d = simulate(n_units=3000, T=5, design=design, seed=11)
    res = _model(d, design).fit()
    true = truth(design, n_big_units=100_000)
    for k, (phi, psi) in enumerate(res.targets):
        err = res.estimates[k, 0] - true[("b", phi, psi)]
        assert abs(err) < 4.5 * res.standard_errors[k, 0] + 0.01, (phi, psi, err, res.standard_errors[k, 0])


def test_one_fold_automatic_equals_plugin():
    d = simulate(n_units=400, design="tvar", seed=3)
    res = _model(d, "tvar").fit(n_folds=1, riesz="automatic")
    np.testing.assert_allclose(res.estimates, res.plugin_estimates, atol=1e-8)
    np.testing.assert_allclose(res.standard_errors, res.plugin_standard_errors, rtol=1e-8)


def test_degree_one_ols_is_fe_ols_with_unit_clustered_se():
    d = simulate(n_units=300, design="none", seed=5)
    res = _model(d).fit(degree=1, interact=False)
    unit = d["frame"]["unit"].to_numpy()
    ly = np.log(d["y"])
    s = d["s"]
    sd = s - pd.Series(s).groupby(unit).transform("mean").to_numpy()
    yd = ly - pd.Series(ly).groupby(unit).transform("mean").to_numpy()
    beta = (sd @ yd) / (sd @ sd)
    e = yd - beta * sd
    k00 = res.targets.index((0.0, 0.0))
    assert res.estimates[k00, 0] == pytest.approx(beta, abs=1e-10)
    # cluster-robust variance with G / (G - 1)
    g = np.bincount(unit, weights=sd * e)
    var = (len(g) / (len(g) - 1)) * np.sum(g ** 2) / (sd @ sd) ** 2
    assert res.standard_errors[k00, 0] == pytest.approx(np.sqrt(var), rel=1e-6)
    # with h = 1, degree 1 and no cells every phi = 0 target is the OLS slope
    k01 = res.targets.index((0.0, 1.0))
    assert res.estimates[k01, 0] == pytest.approx(beta, abs=1e-10)


def test_degree_one_poisson_is_fe_poisson():
    d = simulate(n_units=60, T=4, design="none", seed=2)
    res = _model(d).fit(degree=1, interact=False)
    unit = d["frame"]["unit"].to_numpy()
    dummies = pd.get_dummies(unit).to_numpy(float)
    glm = sm.GLM(d["y"], np.column_stack([d["s"], dummies]), family=sm.families.Poisson()).fit(tol=1e-12)
    k10 = res.targets.index((1.0, 0.0))
    assert res.estimates[k10, 0] == pytest.approx(glm.params[0], abs=1e-6)


def test_reading_b_invariant_to_unit_rescaling_without_cells():
    d = simulate(n_units=400, design="none", seed=8)
    base = _model(d).fit()
    rng = np.random.default_rng(0)
    d2 = dict(d)
    d2["y"] = d["y"] * np.exp(rng.normal(0, 2, 400))[d["frame"]["unit"].to_numpy()]
    moved = _model(d2).fit()
    # FE-OLS is exactly invariant; FE-Poisson weights units by their scale in a
    # finite sample, so phi = 1 is invariant only in the population.
    for k, (phi, psi) in enumerate(base.targets):
        if phi == 0.0:
            assert moved.estimates[k, 0] == pytest.approx(base.estimates[k, 0], abs=1e-8)
        else:
            assert abs(moved.estimates[k, 0] - base.estimates[k, 0]) < 4 * base.standard_errors[k, 0]


def test_explicit_close_to_automatic_at_phi_zero():
    d = simulate(n_units=1500, design="tinv", seed=4)
    m = _model(d, "tinv")
    auto, expl = m.fit(), m.fit(riesz="explicit")
    for k, (phi, psi) in enumerate(auto.targets):
        if phi == 0.0:
            assert abs(auto.estimates[k, 0] - expl.estimates[k, 0]) < 1.5 * auto.standard_errors[k, 0]
    info = expl.diagnostics["folds"][0]["neumann"]
    assert all(v["converged"] for v in info.values())


def test_cross_fitting_time_effects_unbalanced_and_diagnostics():
    d = simulate(n_units=600, design="tvar", seed=6, time_effects=True, unbalanced=0.15)
    m = _model(d, "tvar", time_effects=True)
    res = m.fit(n_folds=3, random_state=1)
    assert np.all(np.isfinite(res.estimates)) and np.all(res.standard_errors > 0)
    assert set(np.unique(res.per_row["fold"])) == {0, 1, 2}
    conn = res.diagnostics["connectedness"]
    assert conn["n_components"] == 1 and conn["n_kappa_cells"] == 5
    assert 0 < res.diagnostics["persistence"]["rho_star"] < 1
    one = m.fit(n_folds=1)
    assert np.all(np.abs(one.estimates - res.estimates) < 6 * one.standard_errors + 0.05)


def test_time_invariant_cells_are_separate_components():
    d = simulate(n_units=500, design="tinv", seed=7)
    m = PanelReadingB(d["y"], d["frame"][["s", "w", "unit"]], fixed_effects=["unit"], unit="unit",
                      interest="s", cells=["w"])
    res = m.fit()
    conn = res.diagnostics["connectedness"]
    assert conn["n_components"] == 2 and len(conn["slope_cells"]) == 2
    assert conn["within_cell_connected"]
    assert np.all(np.isfinite(res.estimates))


def test_pluggable_within_model_matches_sieve_roughly():
    from loglinearcorrection.nonparametric import FESplineModelNuisance

    d = simulate(n_units=800, design="none", seed=9)
    m = _model(d)
    sieve = m.fit()
    spline = m.fit(within_model={"model_class": FESplineModelNuisance, "arch_params": {"n_knots": 5},
                                 "fit_params": {}})
    assert spline.diagnostics["settings"]["within_model"] == "FESplineModelNuisance"
    for k in range(len(sieve.targets)):
        assert abs(spline.estimates[k, 0] - sieve.estimates[k, 0]) < 4 * sieve.standard_errors[k, 0]


def test_weights_and_clusters():
    d = simulate(n_units=300, design="none", seed=12)
    w = np.random.default_rng(1).uniform(0.5, 2.0, len(d["y"]))
    m = _model(d, weights=w)
    res = m.fit()
    region = d["frame"]["unit"].to_numpy() // 10
    coarse = res.with_clusters(region)
    assert coarse.standard_errors.shape == res.standard_errors.shape
    with pytest.warns(UserWarning, match="span several clusters"):
        warnings.simplefilter("always")
        res.with_clusters(np.arange(len(d["y"])) % 7)


# ----------------------------------------------------------------------------- n_folds = 1 elsewhere


def test_stein_target_without_cross_fitting():
    """With mu_hat fitted on the same rows, E[s(v_in) v_in] = -(1 - dof / n)."""
    from loglinearcorrection.generalized_dream import GeneralizedDREAM

    rng = np.random.default_rng(0)
    n_groups, g = 400, 4
    fe = np.repeat(np.arange(n_groups), g)
    x = rng.normal(0, 1, n_groups)[fe] + rng.normal(0, 1, n_groups * g)
    y = np.exp(0.5 * x + rng.normal(0, 0.3, len(x)))
    exog = pd.DataFrame({"x": x, "fe": fe})
    model = GeneralizedDREAM(y, exog, fixed_effects=["fe"], interest="x", fe_encoding="codes")
    rows = np.arange(len(x))
    out = model._partial_out_interest(rows, rows, np.ones(len(x)), list(model.interest), dof_correction=True)
    assert out["stein_target"] == pytest.approx(-(1 - n_groups / len(x)))
    v_in = out["v_test"][:, 0]
    # the true (Gaussian, unit variance) score evaluated at the in-sample residual
    assert np.mean(-v_in * v_in) == pytest.approx(out["stein_target"], abs=0.03)
    held = model._partial_out_interest(rows[::2], rows[1::2], np.ones(len(x)), list(model.interest), True)
    assert held["stein_target"] == -1.0


def test_gpcam_accepts_one_fold():
    from loglinearcorrection.generalized_dream import GeneralizedPCAM
    from loglinearcorrection.nonparametric import FESplineModelNuisance

    rng = np.random.default_rng(1)
    n_units, T = 150, 4
    unit = np.repeat(np.arange(n_units), T)
    c = rng.normal(0, 0.5, n_units)[unit]
    d = (rng.random(len(unit)) < 0.5).astype(float)
    z = rng.normal(0, 1, len(unit))
    y = np.exp(c + 0.3 * d + 0.2 * z + rng.normal(0, 0.4, len(unit)))
    exog = pd.DataFrame({"d": d, "z": z, "unit": unit})
    spec = {"model_class": FESplineModelNuisance, "arch_params": {}, "fit_params": {}}
    model = GeneralizedPCAM(y, exog, fixed_effects=["unit"], treatment="d", phi=1.0, psi=1.0,
                            fe_encoding="codes", seed=1)
    res = model.fit(n_folds=1, m_params=spec, propensity_params=spec)
    assert res.n_folds == 1 and len(res.fold_results) == 1
    assert np.array_equal(res.fold_results[0]["test_idx"], np.arange(len(y)))
    assert np.isfinite(res.estimate) and res.standard_error > 0
    assert abs(res.estimate - np.expm1(0.3)) < 5 * res.standard_error + 0.05


def test_detect_panel_structure():
    from loglinearcorrection.panel_readingb import detect_panel_structure

    rng = np.random.default_rng(0)
    unit = np.repeat(np.arange(200), 5)
    year = np.tile(np.arange(5), 200)
    region_year = (unit // 50) * 10 + year
    x = rng.normal(0, 1, 200)[unit] + rng.normal(0, 1, len(unit))
    fe = pd.DataFrame({"region_year": region_year, "unit": unit, "year": year})
    out = detect_panel_structure(fe, x)
    assert out["unit"] == "unit" and out["time"] == "year" and out["absorbed"] == ["region_year"]
    # an interest regressor constant within units rules the unit out
    out = detect_panel_structure(fe[["unit", "year"]], rng.normal(0, 1, 200)[unit])
    assert out["unit"] == "year"


@pytest.mark.parametrize("kappa_cells", ["crossed", "user", "auto"])
def test_in_sample_corrections_vanish_with_continuous_controls_and_time_effects(kappa_cells):
    """Exact delta-method pieces: with one fold the kappa term and the same-basis
    Riesz term sum to zero, so debiased = plug-in, also with a continuous control
    in the kappa GLM, time cells and (for 'user') absorbed effects outside kappa."""
    d = simulate(n_units=600, design="tvar", seed=3, time_effects=True, unbalanced=0.1)
    frame = d["frame"].copy()
    frame["z"] = np.random.default_rng(0).normal(0, 1, len(frame)) + 0.3 * frame["s"]
    frame["ry"] = (frame["unit"] % 20) * 10 + frame["t"]
    m = PanelReadingB(d["y"], frame[["s", "w", "z", "unit", "t", "ry"]], fixed_effects=["unit", "t", "ry"],
                      unit="unit", interest="s", kappa_cells=kappa_cells)
    res = m.fit()
    w = m.w
    for phi, psi in res.targets:
        assert abs(np.sum(w * res.per_row[f"kappa_correction_{phi:g}_{psi:g}"])) < 1e-8 * w.sum()
    np.testing.assert_allclose(res.estimates, res.plugin_estimates, atol=1e-7)
