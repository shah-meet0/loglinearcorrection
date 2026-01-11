"""
Test suite for IV-DRNO estimator.

Test categories:
1. Unit tests - individual component verification
2. Integration tests - full estimator behavior
3. Orthogonality tests - Neyman orthogonality properties
4. Edge case tests - boundary conditions

Run with: poetry run pytest tests/test_iv_drno.py -v
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from typing import Dict

from conftest import (
    generate_iv_data_heteroskedastic,
    generate_iv_data_homoskedastic,
    IVDGPData,
    DEFAULT_FAST_PARAMS
)


torch = pytest.importorskip("torch")


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestFirstStageRecovery:
    """Test first stage g(Z) estimation."""

    def test_first_stage_predicts_X(self, medium_heteroskedastic_data, fast_fit_params):
        """First stage predictions should correlate with X."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)

        g_hat = data.X.ravel() - results.V_hat.ravel()

        corr = np.corrcoef(g_hat, data.X.ravel())[0, 1]
        assert corr > 0.6, f"First stage correlation {corr:.3f} below threshold"

    def test_V_hat_uncorrelated_with_Z(self, medium_heteroskedastic_data, fast_fit_params):
        """Estimated residual V_hat should be uncorrelated with Z."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)

        corr = np.corrcoef(results.V_hat.ravel(), data.Z.ravel())[0, 1]
        assert abs(corr) < 0.15, f"V_hat-Z correlation {corr:.3f} above threshold"

    def test_V_hat_mean_approximately_zero(self, medium_heteroskedastic_data, fast_fit_params):
        """E[V|Z] = 0 implies E[V_hat] should be close to zero."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)

        mean_V = np.mean(results.V_hat)
        assert abs(mean_V) < 0.2, f"Mean V_hat {mean_V:.3f} far from zero"


class TestControlFunctionOLS:
    """Test control function OLS estimation."""

    def test_beta_recovery(self, medium_heteroskedastic_data, fast_fit_params):
        """Control function beta should recover true beta."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)

        beta_hat = results.beta[0]
        beta_true = data.params.beta

        assert abs(beta_hat - beta_true) < 0.1, \
            f"Beta estimate {beta_hat:.3f} differs from true {beta_true:.3f}"

    def test_rho_recovery(self, medium_heteroskedastic_data, fast_fit_params):
        """Control function rho should recover true rho."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)

        rho_hat = results.rho[0]
        rho_true = data.params.rho

        assert abs(rho_hat - rho_true) < 0.15, \
            f"Rho estimate {rho_hat:.3f} differs from true {rho_true:.3f}"

    def test_endogeneity_detected(self, medium_heteroskedastic_data, fast_fit_params):
        """Control function test should detect endogeneity when rho != 0."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)

        cf_test = results.control_function_test()
        assert cf_test['p_value'] < 0.05, \
            f"Failed to detect endogeneity (p={cf_test['p_value']:.3f})"


class TestOmegaDensityRatio:
    """Test density ratio omega estimation."""

    def test_omega_calibration(self):
        """E[omega(X,V)] should be approximately 1."""
        from loglinearcorrection.iv_nonparametric import NNModelDensityRatio

        np.random.seed(42)
        n = 2000

        V = np.random.randn(n)
        X = 0.5 * V + np.random.randn(n)

        X = X.reshape(-1, 1)
        V = V.reshape(-1, 1)

        omega_model = NNModelDensityRatio(hidden_layers=[64, 64])
        omega_results = omega_model.fit(
            X, V, epochs=50, n_permutations=5, verbose=False
        )

        omega = omega_results.predict(X, V)

        mean_omega = np.mean(omega)
        assert 0.5 < mean_omega < 2.0, \
            f"Mean omega {mean_omega:.3f} outside expected range [0.5, 2.0]"

    def test_omega_bounded(self):
        """Omega values should be bounded."""
        from loglinearcorrection.iv_nonparametric import NNModelDensityRatio

        np.random.seed(42)
        n = 1000

        V = np.random.randn(n)
        X = 0.5 * V + np.random.randn(n)

        omega_model = NNModelDensityRatio(hidden_layers=[32, 32])
        omega_results = omega_model.fit(
            X.reshape(-1, 1), V.reshape(-1, 1),
            epochs=30, n_permutations=3, verbose=False
        )

        omega = omega_results.predict(X.reshape(-1, 1), V.reshape(-1, 1))

        assert omega.min() >= 0.01, f"Omega min {omega.min():.4f} below clip threshold"
        assert omega.max() <= 100.0, f"Omega max {omega.max():.4f} above clip threshold"


class TestLambdaEstimation:
    """Test automatic DML lambda estimation."""

    def test_lambda_learns_conditional_mean(self):
        """Lambda model should learn E[target | Z]."""
        from loglinearcorrection.iv_nonparametric import NNModelRieszLambda

        np.random.seed(42)
        n = 2000

        Z = np.random.randn(n, 2)
        true_lambda = 0.5 * Z[:, 0] + 0.3 * Z[:, 1]
        noise = 0.2 * np.random.randn(n)
        target = true_lambda + noise

        lambda_model = NNModelRieszLambda(hidden_layers=[32, 32])
        lambda_results = lambda_model.fit(Z, target, epochs=100, verbose=False)

        lambda_pred = lambda_results.predict(Z)

        corr = np.corrcoef(lambda_pred, true_lambda)[0, 1]
        assert corr > 0.8, f"Lambda correlation {corr:.3f} below threshold"

        mse = np.mean((lambda_pred - true_lambda)**2)
        signal_var = np.var(true_lambda)
        r2 = 1 - mse / signal_var
        assert r2 > 0.5, f"Lambda R² {r2:.3f} below threshold"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHomoskedasticCase:
    """When Var(eps|X) = constant, semi-elasticity should equal beta."""

    def test_estimate_equals_beta(self, homoskedastic_data, fast_fit_params):
        """Under homoskedasticity, IV-DRNO should estimate beta."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = homoskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)

        if isinstance(results.elasticities, pd.DataFrame):
            estimate = float(results.elasticities['estimate'].iloc[0])
        else:
            estimate = float(results.elasticities[0])

        assert abs(estimate - data.params.beta) < 0.15, \
            f"Estimate {estimate:.3f} differs from beta {data.params.beta:.3f}"

    def test_no_spurious_correction(self, homoskedastic_data, fast_fit_params):
        """IV-DRNO shouldn't add spurious correction under homoskedasticity."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = homoskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)

        if isinstance(results.elasticities, pd.DataFrame):
            estimate = float(results.elasticities['estimate'].iloc[0])
        else:
            estimate = float(results.elasticities[0])

        beta_hat = results.beta[0]

        correction = estimate - beta_hat
        assert abs(correction) < 0.2, \
            f"Spurious correction {correction:.3f} under homoskedasticity"


class TestHeteroskedasticCase:
    """Test IV-DRNO under heteroskedasticity."""

    def test_estimate_direction(self, medium_heteroskedastic_data, fast_fit_params):
        """With a > 0, E[X] > 0: semi-elasticity > beta."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)

        if isinstance(results.elasticities, pd.DataFrame):
            estimate = float(results.elasticities['estimate'].iloc[0])
        else:
            estimate = float(results.elasticities[0])

        beta_hat = results.beta[0]

        assert estimate > beta_hat, \
            f"Estimate {estimate:.3f} should be > beta {beta_hat:.3f}"

    def test_estimate_magnitude(self, medium_heteroskedastic_data, fast_fit_params):
        """IV-DRNO estimate should be close to true semi-elasticity."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=3, random_state=42, **fast_fit_params)

        if isinstance(results.elasticities, pd.DataFrame):
            estimate = float(results.elasticities['estimate'].iloc[0])
        else:
            estimate = float(results.elasticities[0])

        bias = abs(estimate - data.true_semi_elasticity)
        assert bias < 0.15, \
            f"Bias {bias:.3f} (estimate={estimate:.3f}, true={data.true_semi_elasticity:.3f})"

    def test_beats_2sls(self, medium_heteroskedastic_data, fast_fit_params):
        """IV-DRNO should be closer to true semi-elasticity than 2SLS."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data

        Z_const = sm.add_constant(data.Z)
        first_stage = sm.OLS(data.X.ravel(), Z_const).fit()
        X_hat = first_stage.fittedvalues.reshape(-1, 1)
        X_hat_const = sm.add_constant(X_hat)
        tsls = sm.OLS(np.log(data.Y), X_hat_const).fit()
        tsls_estimate = tsls.params[1]

        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])
        results = model.fit(n_folds=3, random_state=42, **fast_fit_params)

        if isinstance(results.elasticities, pd.DataFrame):
            ivdrno_estimate = float(results.elasticities['estimate'].iloc[0])
        else:
            ivdrno_estimate = float(results.elasticities[0])

        tsls_gap = abs(tsls_estimate - data.true_semi_elasticity)
        ivdrno_gap = abs(ivdrno_estimate - data.true_semi_elasticity)

        assert ivdrno_gap < tsls_gap, \
            f"IV-DRNO gap {ivdrno_gap:.3f} should be < 2SLS gap {tsls_gap:.3f}"

    def test_confidence_interval_reasonable(self, medium_heteroskedastic_data, fast_fit_params):
        """95% CI should have reasonable width and estimate near true value."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=3, random_state=42, **fast_fit_params)

        ci = list(results.confidence_intervals.values())[0]
        lower, upper = ci
        width = upper - lower

        if isinstance(results.elasticities, pd.DataFrame):
            estimate = float(results.elasticities['estimate'].iloc[0])
        else:
            estimate = float(results.elasticities[0])

        # CI width should be reasonable (not too narrow or too wide)
        assert 0.05 < width < 1.0, \
            f"CI width {width:.3f} outside reasonable range [0.05, 1.0]"

        # CI should be centered near the estimate
        ci_center = (lower + upper) / 2
        assert abs(ci_center - estimate) < 0.01, \
            f"CI center {ci_center:.3f} far from estimate {estimate:.3f}"

        # Estimate should be reasonably close to true value
        bias = abs(estimate - data.true_semi_elasticity)
        assert bias < 0.2, \
            f"Bias {bias:.3f} too large (estimate={estimate:.3f}, true={data.true_semi_elasticity:.3f})"


class TestInstrumentStrength:
    """Test behavior with different instrument strengths."""

    def test_strong_instrument(self, strong_instrument_data, fast_fit_params):
        """With strong instrument, estimates should have low bias."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = strong_instrument_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=3, random_state=42, **fast_fit_params)

        if isinstance(results.elasticities, pd.DataFrame):
            estimate = float(results.elasticities['estimate'].iloc[0])
        else:
            estimate = float(results.elasticities[0])

        bias = abs(estimate - data.true_semi_elasticity)
        assert bias < 0.12, f"Strong instrument bias {bias:.3f}"

    def test_weak_instrument(self, weak_instrument_data, fast_fit_params):
        """With weak instrument, estimates may have higher variance."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = weak_instrument_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=3, random_state=42, **fast_fit_params)

        if isinstance(results.elasticities, pd.DataFrame):
            estimate = float(results.elasticities['estimate'].iloc[0])
        else:
            estimate = float(results.elasticities[0])

        bias = abs(estimate - data.true_semi_elasticity)
        assert bias < 0.4, f"Weak instrument bias {bias:.3f}"


class TestCrossFitting:
    """Test cross-fitting behavior."""

    def test_different_folds_similar_results(self, medium_heteroskedastic_data, fast_fit_params):
        """Estimates with 2, 3, 5 folds should be similar."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data

        estimates = []
        for n_folds in [2, 3, 5]:
            model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])
            results = model.fit(n_folds=n_folds, random_state=42, **fast_fit_params)

            if isinstance(results.elasticities, pd.DataFrame):
                est = float(results.elasticities['estimate'].iloc[0])
            else:
                est = float(results.elasticities[0])
            estimates.append(est)

        max_diff = max(estimates) - min(estimates)
        assert max_diff < 0.15, \
            f"Estimates vary across folds: {estimates}, diff={max_diff:.3f}"


# =============================================================================
# ORTHOGONALITY TESTS
# =============================================================================

class TestNeymanOrthogonality:
    """Test Neyman orthogonality properties."""

    def test_moments_mean_equals_estimate(self, medium_heteroskedastic_data, fast_fit_params):
        """Mean of moments should equal the estimate."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=3, random_state=42, **fast_fit_params)

        moments_mean = np.mean(results.moments)

        if isinstance(results.elasticities, pd.DataFrame):
            estimate = float(results.elasticities['estimate'].iloc[0])
        else:
            estimate = float(results.elasticities[0])

        assert abs(moments_mean - estimate) < 0.1, \
            f"Moments mean {moments_mean:.3f} differs from estimate {estimate:.3f}"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestReproducibility:
    """Test seed reproducibility."""

    def test_same_seed_same_results(self, small_heteroskedastic_data, fast_fit_params):
        """Two runs with same seed should give identical results."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = small_heteroskedastic_data

        estimates = []
        for _ in range(2):
            # Set all random seeds for reproducibility
            np.random.seed(123)
            torch.manual_seed(123)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(123)

            model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])
            results = model.fit(n_folds=2, random_state=123, **fast_fit_params)

            if isinstance(results.elasticities, pd.DataFrame):
                est = float(results.elasticities['estimate'].iloc[0])
            else:
                est = float(results.elasticities[0])
            estimates.append(est)

        assert abs(estimates[0] - estimates[1]) < 1e-6, \
            f"Same seed gave different results: {estimates}"

    def test_different_seeds_different_results(self, small_heteroskedastic_data, fast_fit_params):
        """Different seeds should give different results."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = small_heteroskedastic_data

        estimates = []
        for seed in [42, 123]:
            model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])
            results = model.fit(n_folds=2, random_state=seed, **fast_fit_params)

            if isinstance(results.elasticities, pd.DataFrame):
                est = float(results.elasticities['estimate'].iloc[0])
            else:
                est = float(results.elasticities[0])
            estimates.append(est)

        assert abs(estimates[0] - estimates[1]) > 1e-8, \
            "Different seeds gave identical results"


class TestInputFormats:
    """Test input format handling."""

    def test_numpy_arrays(self, small_heteroskedastic_data, fast_fit_params):
        """Should work with numpy arrays."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = small_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)
        assert results is not None

    def test_1d_arrays(self, small_heteroskedastic_data, fast_fit_params):
        """Should handle 1D arrays for Y."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = small_heteroskedastic_data

        model = IVDREEM(
            endog=data.Y.ravel(),
            exog=data.X,
            instruments=data.Z,
            interest=[0]
        )

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)
        assert results is not None


class TestMultipleEndogenous:
    """Test with multiple endogenous variables."""

    def test_two_endogenous(self, fast_fit_params):
        """Should handle k_x = 2 case."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        np.random.seed(42)
        n = 2000

        Z = np.random.randn(n, 2)
        V = np.random.randn(n, 2)

        Gamma = np.array([[1.0, 0.3], [0.2, 1.0]])
        X = Z @ Gamma + V

        beta = np.array([0.5, -0.3])
        rho = np.array([0.8, 0.6])
        eps = 0.5 * np.random.randn(n)

        log_Y = X @ beta + V @ rho + eps
        Y = np.exp(log_Y)

        model = IVDREEM(endog=Y, exog=X, instruments=Z, interest=[0, 1])

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)

        if isinstance(results.elasticities, pd.DataFrame):
            n_estimates = len(results.elasticities)
        else:
            n_estimates = len(results.elasticities)

        assert n_estimates == 2, f"Expected 2 estimates, got {n_estimates}"


class TestExogenousControls:
    """Test with exogenous control variables."""

    def test_with_controls(self, fast_fit_params):
        """Should handle exogenous controls W."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        np.random.seed(42)
        n = 2000

        Z = np.random.randn(n, 1)
        W = np.random.randn(n, 2)
        V = np.random.randn(n)

        X = Z.ravel() + 0.5 * W[:, 0] + V
        beta = 0.5
        gamma = np.array([0.3, -0.2])
        rho = 0.8
        eps = 0.5 * np.random.randn(n)

        log_Y = beta * X + W @ gamma + rho * V + eps
        Y = np.exp(log_Y)

        model = IVDREEM(
            endog=Y,
            exog=X.reshape(-1, 1),
            instruments=Z,
            exog_control=W,
            interest=[0]
        )

        results = model.fit(n_folds=2, random_state=42, **fast_fit_params)

        assert results.gamma is not None, "Expected gamma estimates for controls"
        assert len(results.gamma) == 2, f"Expected 2 gamma estimates, got {len(results.gamma)}"


# =============================================================================
# SLOW TESTS
# =============================================================================

@pytest.mark.slow
class TestLargeSample:
    """Large sample tests."""

    def test_asymptotic_bias(self, large_heteroskedastic_data, fast_fit_params):
        """With large n, bias should be small."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = large_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=5, random_state=42, **fast_fit_params)

        if isinstance(results.elasticities, pd.DataFrame):
            estimate = float(results.elasticities['estimate'].iloc[0])
        else:
            estimate = float(results.elasticities[0])

        bias = abs(estimate - data.true_semi_elasticity)
        assert bias < 0.08, f"Large sample bias {bias:.3f}"


class TestArchitectureSensitivity:
    """Test sensitivity to network architecture."""

    def test_different_hidden_layers(self, small_heteroskedastic_data):
        """Results should be similar with different network sizes."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = small_heteroskedastic_data

        estimates = []
        for hidden_size in [32, 64, 128]:
            params = {
                'first_stage_params': {
                    'arch_params': {'hidden_layers': [hidden_size, hidden_size]},
                    'fit_params': {'epochs': 30, 'verbose': False}
                },
                'm_params': {
                    'arch_params': {'hidden_layers': [hidden_size, hidden_size]},
                    'fit_params': {'epochs': 50, 'verbose': False}
                },
                'density_params': {
                    'arch_params': {'shared': {'hidden_layers': [hidden_size, hidden_size]}},
                    'fit_params': {'epochs': 50, 'verbose': False}
                },
                'omega_params': {
                    'arch_params': {'hidden_layers': [hidden_size // 2, hidden_size // 2]},
                    'fit_params': {'epochs': 30, 'n_permutations': 3, 'verbose': False}
                },
                'lambda_params': {
                    'arch_params': {'hidden_layers': [hidden_size // 4, hidden_size // 4]},
                    'fit_params': {'epochs': 20, 'verbose': False}
                },
                'n_mc_samples': 200
            }

            model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])
            results = model.fit(n_folds=2, random_state=42, **params)

            if isinstance(results.elasticities, pd.DataFrame):
                est = float(results.elasticities['estimate'].iloc[0])
            else:
                est = float(results.elasticities[0])
            estimates.append(est)

        max_diff = max(estimates) - min(estimates)
        assert max_diff < 0.25, \
            f"Estimates vary with architecture: {estimates}, diff={max_diff:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
