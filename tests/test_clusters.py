"""
Tests for cluster-robust covariances built from influence functions.
"""
import numpy as np
import pytest
import statsmodels.api as sm

from loglinearcorrection.hypothesis import InfluenceFunctionDiffTest, influence_covariance


def _ols_data(seed=0, n=600, weighted=False):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 30, n)
    b = rng.integers(0, 12, n)
    x = rng.normal(size=n) + rng.normal(size=30)[a]
    e = rng.normal(size=n) + rng.normal(size=30)[a] + rng.normal(size=12)[b]
    X = np.column_stack([np.ones(n), x])
    y = 1.0 + 0.5 * x + e
    w = rng.uniform(0.5, 2.0, n) if weighted else np.ones(n)
    return X, y, w, np.column_stack([a, b])


def _ols_influence(X, y, w):
    fit = sm.WLS(y, X, weights=w).fit()
    p = w / w.sum()
    bread = (X.T * p) @ X
    return fit, (X @ np.linalg.inv(bread)) * fit.resid[:, None], p


@pytest.mark.parametrize("weighted", [False, True])
def test_one_way_matches_statsmodels(weighted):
    X, y, w, groups = _ols_data(weighted=weighted)
    _, influence, p = _ols_influence(X, y, w)
    expected = sm.WLS(y, X, weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": groups[:, 0], "use_correction": False}
    ).cov_params()
    got = influence_covariance(influence, p, groups[:, 0], small_sample=False)
    np.testing.assert_allclose(got, expected, rtol=1e-8)


def test_two_way_matches_statsmodels():
    X, y, w, groups = _ols_data(seed=1)
    _, influence, p = _ols_influence(X, y, w)
    expected = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": groups, "use_correction": False}
    ).cov_params()
    got = influence_covariance(influence, p, groups, small_sample=False)
    np.testing.assert_allclose(got, expected, rtol=1e-8)


def test_no_clusters_and_singleton_clusters_are_row_level():
    X, y, w, _ = _ols_data(seed=2)
    _, influence, p = _ols_influence(X, y, w)
    row = influence_covariance(influence, p)
    np.testing.assert_allclose(row, sm.OLS(y, X).fit(cov_type="HC0").cov_params(), rtol=1e-8)
    singletons = influence_covariance(influence, p, np.arange(len(y)), small_sample=False)
    np.testing.assert_allclose(singletons, row, rtol=1e-10)


def test_small_sample_factor_and_string_labels():
    X, y, w, groups = _ols_data(seed=3)
    _, influence, p = _ols_influence(X, y, w)
    labels = np.array([f"g{g}" for g in groups[:, 0]])
    plain = influence_covariance(influence, p, labels, small_sample=False)
    corrected = influence_covariance(influence, p, labels)
    n_groups = len(np.unique(labels))
    np.testing.assert_allclose(corrected, plain * n_groups / (n_groups - 1), rtol=1e-12)


def test_diff_test_uses_clusters():
    X, y, w, groups = _ols_data(seed=4)
    fit, influence, _ = _ols_influence(X, y, w)
    row = InfluenceFunctionDiffTest(n=len(y), k=1, coefs=fit.params, influence_functions=influence)
    clustered = InfluenceFunctionDiffTest(n=len(y), k=1, coefs=fit.params,
                                          influence_functions=influence, clusters=groups[:, 0])
    assert clustered.results[0].standard_error > row.results[0].standard_error
    with pytest.raises(ValueError):
        influence_covariance(influence, np.full(len(y), 1 / len(y)), groups[:10, 0])
