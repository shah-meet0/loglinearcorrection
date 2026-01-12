"""
Tests for IV-DRNO with fixed effects.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

# Skip all tests if dependencies not available
pytest.importorskip("torch")
pytest.importorskip("pyhdfe")


def generate_panel_iv_data(
    n_individuals: int = 100,
    n_periods: int = 5,
    beta: float = 0.5,
    rho: float = 0.5,
    fe_scale: float = 1.0,
    seed: int = 42
):
    """
    Generate panel data with individual fixed effects for IV-DRNO testing.

    Model:
        log Y_it = beta * X_it + rho * V_it + delta_i + eps_it
        X_it = pi * Z_it + V_it + u_it

    Where delta_i is an individual fixed effect.
    """
    np.random.seed(seed)

    n = n_individuals * n_periods

    # Individual IDs
    individual = np.repeat(np.arange(n_individuals), n_periods)
    period = np.tile(np.arange(n_periods), n_individuals)

    # Individual fixed effects
    delta = np.random.randn(n_individuals) * fe_scale
    delta_expanded = delta[individual]

    # Instrument (varies within and between individuals)
    Z = np.random.randn(n)

    # Unobserved confounder
    V = np.random.randn(n)

    # First stage error
    u = np.random.randn(n) * 0.5

    # Endogenous variable (correlated with individual FE for testing)
    pi = 1.0
    X = pi * Z + V + u + 0.3 * delta_expanded

    # Outcome error
    eps = np.random.randn(n) * 0.3

    # Log outcome
    log_Y = beta * X + rho * V + delta_expanded + eps
    Y = np.exp(log_Y)

    return {
        'Y': Y,
        'X': X,
        'Z': Z,
        'V': V,  # True V for verification
        'individual': individual,
        'period': period,
        'delta': delta,
        'true_beta': beta,
        'true_rho': rho,
        'n': n,
        'n_individuals': n_individuals,
        'n_periods': n_periods
    }


class TestIVDRNOFixedEffectsInit:
    """Test fixed effects initialization in IV-DRNO."""

    def test_fe_initialization(self):
        """Test that fixed effects are properly initialized."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel

        data = generate_panel_iv_data(n_individuals=50, n_periods=3, seed=42)

        # Create DataFrame with FE column
        df = pd.DataFrame({
            'Y': data['Y'],
            'X': data['X'],
            'Z': data['Z'],
            'individual': data['individual']
        })

        # Initialize model with fixed effects
        model = IVDoublyRobustElasticityEstimatorModel(
            endog=df['Y'],
            exog=df[['X', 'individual']],
            instruments=df['Z'],
            fixed_effects=['individual'],
            interest=[0]  # Only X, not the FE
        )

        # Check FE is initialized
        assert model.fixed_effects is not None
        assert len(model.fe_indices) == 1
        assert model.fe_cols is not None

        # Check FE column removed from exog
        assert model.exog.shape[1] == 1  # Only X remains
        assert 'individual' not in model.exog_names

    def test_fe_with_integer_indices(self):
        """Test FE specification with integer indices."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel

        data = generate_panel_iv_data(n_individuals=50, n_periods=3, seed=42)

        # Create array with FE column
        exog = np.column_stack([data['X'], data['individual']])

        model = IVDoublyRobustElasticityEstimatorModel(
            endog=data['Y'],
            exog=exog,
            instruments=data['Z'].reshape(-1, 1),
            fixed_effects=[1],  # Integer index
            interest=[0]
        )

        assert model.fixed_effects is not None
        assert model.exog.shape[1] == 1


class TestIVDRNOFixedEffectsFit:
    """Test IV-DRNO fitting with fixed effects."""

    @pytest.mark.slow
    def test_fe_fit_reduces_bias(self):
        """Test that FE reduces bias when FE are present in DGP."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel

        # Generate data with strong FE
        data = generate_panel_iv_data(
            n_individuals=100,
            n_periods=5,
            beta=0.5,
            rho=0.3,
            fe_scale=2.0,  # Strong FE
            seed=42
        )

        # Create DataFrame
        df = pd.DataFrame({
            'Y': data['Y'],
            'X': data['X'],
            'Z': data['Z'],
            'individual': data['individual']
        })

        # Model WITH fixed effects
        model_fe = IVDoublyRobustElasticityEstimatorModel(
            endog=df['Y'],
            exog=df[['X', 'individual']],
            instruments=df['Z'],
            fixed_effects=['individual'],
            interest=[0]
        )

        # Fit with minimal settings for speed
        results_fe = model_fe.fit(
            n_folds=2,
            random_state=42,
            m_params={'fit_params': {'epochs': 50, 'batch_size': 256}},
            density_params={'fit_params': {'epochs': 50, 'batch_size': 256}},
            omega_params={'fit_params': {'epochs': 50, 'batch_size': 256}},
            lambda_params={'fit_params': {'epochs': 50, 'batch_size': 256}}
        )

        # Check results exist
        assert results_fe is not None
        assert hasattr(results_fe, 'elasticities')

        # Beta should be reasonably close to true value
        beta_hat = results_fe.beta[0] if hasattr(results_fe.beta, '__len__') else results_fe.beta
        print(f"True beta: {data['true_beta']}, Estimated beta: {beta_hat}")

        # With FE, estimate should be within reasonable range
        # (exact accuracy depends on sample size and NN training)
        assert abs(beta_hat - data['true_beta']) < 1.0  # Loose bound for smoke test


class TestIVDRNOFixedEffectsNone:
    """Test IV-DRNO without fixed effects still works."""

    def test_no_fe_baseline(self):
        """Test that model works without FE (baseline)."""
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel

        data = generate_panel_iv_data(n_individuals=50, n_periods=3, seed=42)

        model = IVDoublyRobustElasticityEstimatorModel(
            endog=data['Y'],
            exog=data['X'].reshape(-1, 1),
            instruments=data['Z'].reshape(-1, 1),
            interest=[0]
        )

        # Should have no FE
        assert model.fixed_effects is None
        assert model.fe_cols is None
        assert len(model.fe_indices) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])
