"""
Tests for FENNModelNuisance: the neural nuisance with additive fixed-effect scalars.
"""
import numpy as np
import pandas as pd
import pytest

from loglinearcorrection.generalized_dream import GeneralizedPCAM, JointGeneralizedDREAM
from loglinearcorrection.nonparametric import FENNModelNuisance, FENNModelNuisanceResults

FAST = dict(hidden_layers=[32, 32], batch_size=256, epochs=400, patience=20,
            learning_rate=3e-3, device="cpu")


def _data(seed=0, n=3000, n_levels=100, link="identity"):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    g = rng.integers(0, n_levels, n)
    a = rng.normal(size=n_levels)
    f = np.sin(x[:, 0]) + 0.3 * x[:, 1] ** 2
    if link == "identity":
        y = f + a[g] + 0.3 * rng.normal(size=n)
        true = np.cos(x[:, 0])
    elif link == "log":
        mu = np.exp(0.5 * f + 0.3 * a[g])
        y = mu * np.exp(0.3 * rng.normal(size=n) - 0.045)
        true = mu * 0.5 * np.cos(x[:, 0])
    else:
        p = 1.0 / (1.0 + np.exp(-(f - 0.3 + a[g])))
        y = (rng.random(n) < p).astype(float)
        true = p * (1 - p) * np.cos(x[:, 0])
    return x, g, y, true


@pytest.mark.parametrize("link,min_corr", [("identity", 0.97), ("log", 0.9), ("logit", 0.8)])
@pytest.mark.parametrize("with_fe", [True, False])
def test_derivative_recovery(link, min_corr, with_fe):
    # Binary outcomes carry less information per row, so fewer (larger) FE levels there
    n_levels = (30 if link == "logit" else 100) if with_fe else 1
    x, g, y, true = _data(link=link, n_levels=n_levels)
    X = np.column_stack([x, g]) if with_fe else x
    model = FENNModelNuisance({}, non_fe_inputs=2 if with_fe else None,
                              categorical_cardinalities=[n_levels] if with_fe else None, link=link, **FAST)
    result = model.fit(X, y, random_state=0)
    assert isinstance(result, FENNModelNuisanceResults)
    pred, grad = result.derivative(X, [0])
    assert pred.shape == (len(y),) and grad.shape == (len(y), 1)
    np.testing.assert_allclose(pred, result.predict(X), rtol=1e-6, atol=1e-8)
    assert np.corrcoef(grad[:, 0], true)[0, 1] > min_corr
    if link == "log":
        assert np.all(pred > 0)
    if link == "logit":
        assert np.all((pred > 0) & (pred < 1))


def test_derivative_is_raw_scale_autograd():
    # Rescaling a column by 10 must divide its derivative by 10
    x, g, y, _ = _data(n=1500, n_levels=30)
    X = np.column_stack([x, g])
    kw = dict(non_fe_inputs=2, categorical_cardinalities=[30], **FAST)
    pred, grad = FENNModelNuisance({}, **kw).fit(X, y, random_state=3).derivative(X, [0, 1])
    X10 = X.copy()
    X10[:, 0] *= 10.0
    pred10, grad10 = FENNModelNuisance({}, **kw).fit(X10, y, random_state=3).derivative(X10, [0, 1])
    np.testing.assert_allclose(pred10, pred, atol=1e-4)
    np.testing.assert_allclose(grad10[:, 0] * 10.0, grad[:, 0], atol=1e-3)
    # And it matches central finite differences of predict()
    result = FENNModelNuisance({}, link="log", **kw).fit(X, np.exp(0.2 * y), random_state=3)
    _, g_ad = result.derivative(X[:200], [0])
    h = 1e-3
    Xp, Xm = X[:200].copy(), X[:200].copy()
    Xp[:, 0] += h
    Xm[:, 0] -= h
    g_fd = (result.predict(Xp) - result.predict(Xm)) / (2 * h)
    np.testing.assert_allclose(g_ad[:, 0], g_fd, rtol=2e-2, atol=1e-3)
    with pytest.raises(ValueError):
        result.derivative(X, [2])  # FE code column


def test_fe_scalars_unseen_levels_and_weights():
    x, g, y, _ = _data(n=2000, n_levels=50)
    X = np.column_stack([x, g])
    w = np.random.default_rng(1).uniform(0.5, 2.0, len(y))
    model = FENNModelNuisance({}, non_fe_inputs=2, categorical_cardinalities=[60], **FAST)
    result = model.fit(X, y, sample_weight=w, random_state=0)
    alpha = model.net_.alphas[0].detach().numpy()
    assert np.all(alpha[50:] == 0.0)  # levels without training rows
    # Weighted mean of the fitted FE scalars is zero, so unseen codes map to the average level
    assert abs(np.average(alpha[g], weights=w)) < 1e-5
    X_unseen = X[:5].copy()
    X_unseen[:, 2] = 55
    X_out = X[:5].copy()
    X_out[:, 2] = 1000
    np.testing.assert_allclose(result.predict(X_unseen), result.predict(X_out), atol=1e-6)


def test_deterministic_given_seed_and_minibatch():
    x, g, y, _ = _data(n=1500, n_levels=20)
    X = np.column_stack([x, g])
    kw = dict(non_fe_inputs=2, categorical_cardinalities=[20], hidden_layers=[16], batch_size=128,
              epochs=40, patience=5, dropout=0.1, device="cpu")
    a = FENNModelNuisance({}, **kw).fit(X, y, random_state=7).derivative(X, [0])
    b = FENNModelNuisance({}, **kw).fit(X, y, random_state=7).derivative(X, [0])
    c = FENNModelNuisance({}, **kw).fit(X, y, random_state=8).derivative(X, [0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.allclose(a[1], c[1])
    # Fit-time keys override the constructor's
    r = FENNModelNuisance({}, **kw).fit(X, y, random_state=7, epochs=3, val_frac=0.0)
    assert r.metrics["best_epoch"] == 3


def test_link_defaults_follow_estimator_flags():
    assert FENNModelNuisance({}, positive_log_link=True, log_link_max_abs=3.0, input_size=3).link == "log"
    assert FENNModelNuisance({}, output_activation="sigmoid").link == "logit"
    assert FENNModelNuisance({}, output_activation="identity").link == "identity"
    with pytest.raises(ValueError):
        FENNModelNuisance({}, positive_log_link=True, link="identity")


def _panel(seed=0, n=1200, n_levels=40, binary=False):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_levels, n)
    z = rng.normal(size=n)
    if binary:
        x = (rng.random(n) < 1 / (1 + np.exp(-0.5 * z))).astype(float)
    else:
        x = rng.normal(size=n) + 0.3 * z
    log_y = 0.5 * x + 0.3 * z + rng.normal(size=n_levels)[firm] + 0.5 * rng.normal(size=n)
    exog = pd.DataFrame({"x": x, "z": z, "firm": firm})
    return np.exp(log_y), exog


NN_M = {"model_class": FENNModelNuisance,
        "arch_params": {"hidden_layers": [16, 16], "device": "cpu"},
        "fit_params": {"epochs": 200, "patience": 10, "batch_size": None, "learning_rate": 3e-3}}
SCORE = {"score_method": "partial_out", "score_calibration": "stein",
         "arch_params": {"shared": {"hidden_layers": [16], "activation": "silu", "dropout": 0.0}},
         "fit_params": {"epochs": 30, "batch_size": 256, "learning_rate": 1e-3, "patience": 5,
                        "verbose": False}}


@pytest.mark.parametrize("n_folds", [5, 1])
def test_geam_end_to_end(n_folds):
    y, exog = _panel()
    targets = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))

    def run():
        model = JointGeneralizedDREAM(y, exog, fixed_effects=["firm"], interest=["x"], targets=targets,
                                      fe_encoding="codes", nuisance_fixed_effects=20, seed=1)
        return model.fit(n_folds=n_folds, m_params=NN_M, density_params=SCORE)

    result = run()
    estimates = np.asarray(result.estimates)[:, 0]
    assert estimates.shape == (4,) and np.all(np.isfinite(estimates))
    assert np.all(np.abs(estimates - 0.5) < 0.3)
    assert np.all(np.isfinite(np.asarray(result.influence_functions)))
    assert len(result.fold_results) == n_folds
    np.testing.assert_array_equal(np.asarray(run().estimates), np.asarray(result.estimates))


@pytest.mark.parametrize("n_folds", [5, 1])
@pytest.mark.parametrize("phi,psi", [(0.0, 0.0), (1.0, 1.0)])
def test_gpcam_end_to_end(n_folds, phi, psi):
    y, exog = _panel(binary=True)
    propensity = {"model_class": FENNModelNuisance,
                  "arch_params": {"hidden_layers": [16], "device": "cpu"},
                  "fit_params": {"epochs": 200, "patience": 10, "batch_size": None, "learning_rate": 3e-3}}

    def run():
        model = GeneralizedPCAM(y, exog, fixed_effects=["firm"], treatment="x", phi=phi, psi=psi,
                                fe_encoding="codes", nuisance_fixed_effects=20, seed=2)
        return model.fit(n_folds=n_folds, m_params=NN_M, propensity_params=propensity)

    result = run()
    assert np.isfinite(result.estimate) and np.isfinite(result.standard_error)
    # GPCAM targets a percentage change; the log effect is 0.5
    assert abs(np.log1p(result.estimate) - 0.5) < 0.3 if psi == 1.0 else abs(result.estimate - 0.5) < 0.3
    assert len(result.fold_results) == n_folds
    assert result.estimate == run().estimate


def test_max_seconds_caps_training():
    x, g, y, _ = _data(n=3000, n_levels=30)
    X = np.column_stack([x, g])
    kw = dict(non_fe_inputs=2, categorical_cardinalities=[30], hidden_layers=[64, 64], batch_size=32,
              epochs=10_000, patience=10_000, device="cpu")
    result = FENNModelNuisance({}, **kw).fit(X, y, random_state=0, max_seconds=1.0)
    assert result.metrics["timed_out"] and not result.metrics["refitted"]
    assert result.metrics["seconds"] < 10
    assert np.all(np.isfinite(result.predict(X)))
