"""
Test suite for IV-DRNO linear first stage option.

Tests the linear/2SLS first stage as an alternative to neural networks.

Run with: poetry run pytest tests/test_iv_linear_first_stage.py -v
"""

import numpy as np
import pytest

from conftest import (
    generate_iv_data_heteroskedastic,
    generate_iv_data_homoskedastic,
    IVDGPData,
    DEFAULT_FAST_PARAMS
)

torch = pytest.importorskip("torch")


class TestLinearFirstStageBasic:
    """Basic tests for linear first stage functionality."""

    def test_linear_first_stage_runs(self, medium_heteroskedastic_data, fast_fit_params):
        """Linear first stage runs without errors."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42,
                           first_stage_method='linear', **fast_fit_params)

        assert results is not None
        assert hasattr(results, 'elasticities')
        assert len(results.elasticities) == 1

    def test_2sls_alias_works(self, small_heteroskedastic_data, fast_fit_params):
        """'2sls' works as alias for 'linear'."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = small_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42,
                           first_stage_method='2sls', **fast_fit_params)

        assert results is not None

    def test_invalid_method_raises(self, small_heteroskedastic_data, fast_fit_params):
        """Invalid first_stage_method raises ValueError."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = small_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        with pytest.raises(ValueError, match="Unknown first_stage_method"):
            model.fit(n_folds=2, first_stage_method='invalid', **fast_fit_params)


class TestLinearFirstStageResults:
    """Test quality of linear first stage results."""

    def test_V_hat_uncorrelated_with_Z(self, medium_heteroskedastic_data, fast_fit_params):
        """Linear first stage residuals should be uncorrelated with Z."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42,
                           first_stage_method='linear', **fast_fit_params)

        corr = np.corrcoef(results.V_hat.ravel(), data.Z.ravel())[0, 1]
        # Linear OLS should give exactly orthogonal residuals (up to numerical error)
        assert abs(corr) < 0.05, f"V_hat-Z correlation {corr:.3f} above threshold"

    def test_beta_recovery_linear(self, medium_heteroskedastic_data, fast_fit_params):
        """Linear first stage should recover beta reasonably well."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data
        model = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42,
                           first_stage_method='linear', **fast_fit_params)

        beta_hat = results.beta[0]
        beta_true = data.params.beta

        # Linear first stage should still recover beta reasonably
        assert abs(beta_hat - beta_true) < 0.15, \
            f"Beta estimate {beta_hat:.3f} differs from true {beta_true:.3f}"


class TestLinearVsNNComparison:
    """Compare linear and NN first stage results."""

    def test_linear_vs_nn_similar_on_linear_dgp(self, fast_fit_params):
        """Linear and NN give similar estimates when DGP is actually linear."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        # Generate data with linear first stage (X = gamma*Z + V)
        data = generate_iv_data_homoskedastic(n=3000, seed=42)

        # Fit with NN
        model_nn = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])
        results_nn = model_nn.fit(n_folds=2, random_state=42,
                                  first_stage_method='nn', **fast_fit_params)

        # Fit with linear
        model_linear = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])
        results_linear = model_linear.fit(n_folds=2, random_state=42,
                                          first_stage_method='linear', **fast_fit_params)

        # Estimates should be similar
        est_nn = results_nn.elasticities[0]
        est_linear = results_linear.elasticities[0]

        assert abs(est_nn - est_linear) < 0.3, \
            f"NN estimate {est_nn:.3f} differs from linear {est_linear:.3f}"

    def test_linear_faster_than_nn(self, medium_heteroskedastic_data, fast_fit_params):
        """Linear first stage should be faster than NN (at least not slower)."""
        import time
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        data = medium_heteroskedastic_data

        # Time NN
        model_nn = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])
        start_nn = time.time()
        model_nn.fit(n_folds=2, random_state=42, first_stage_method='nn', **fast_fit_params)
        time_nn = time.time() - start_nn

        # Time linear
        model_linear = IVDREEM(endog=data.Y, exog=data.X, instruments=data.Z, interest=[0])
        start_linear = time.time()
        model_linear.fit(n_folds=2, random_state=42, first_stage_method='linear', **fast_fit_params)
        time_linear = time.time() - start_linear

        # Linear should not be dramatically slower (allow 50% overhead for safety)
        # In practice it should be faster due to no NN training
        assert time_linear < time_nn * 1.5, \
            f"Linear time {time_linear:.2f}s much slower than NN {time_nn:.2f}s"


class TestOveridentification:
    """Test linear first stage with over-identification."""

    def test_overidentified_linear(self, fast_fit_params):
        """Linear handles k_z > k_x (over-identified) correctly."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM

        np.random.seed(42)
        n = 2000

        # 3 instruments for 1 endogenous variable
        Z1 = np.random.randn(n)
        Z2 = np.random.randn(n)
        Z3 = np.random.randn(n)
        Z = np.column_stack([Z1, Z2, Z3])

        V = np.random.randn(n)
        X = 0.5 * Z1 + 0.3 * Z2 + 0.2 * Z3 + V
        eps = 0.5 * np.random.randn(n)
        Y = np.exp(0.5 * X + 0.8 * V + eps)

        model = IVDREEM(endog=Y, exog=X.reshape(-1, 1), instruments=Z, interest=[0])

        results = model.fit(n_folds=2, random_state=42,
                           first_stage_method='linear', **fast_fit_params)

        assert results is not None
        assert len(results.elasticities) == 1
        # Should still get reasonable estimate
        assert abs(results.beta[0] - 0.5) < 0.2


@pytest.fixture
def small_heteroskedastic_data() -> IVDGPData:
    """Small dataset for fast tests."""
    return generate_iv_data_heteroskedastic(n=1000, seed=42)


@pytest.fixture
def medium_heteroskedastic_data() -> IVDGPData:
    """Medium dataset for standard tests."""
    return generate_iv_data_heteroskedastic(n=3000, seed=42)


@pytest.fixture
def fast_fit_params():
    """Fast neural network parameters for quick tests."""
    return DEFAULT_FAST_PARAMS.copy()
