"""
Pytest configuration and shared fixtures for IV-DRNO tests.
"""

import numpy as np
import pytest
from typing import Dict, Tuple, NamedTuple


class IVDGPParams(NamedTuple):
    """Parameters for IV DGP."""
    n: int
    beta: float
    rho: float
    gamma_z: float
    mu_z: float
    sigma_v: float
    sigma_eps: float
    a_hetero: float
    seed: int


class IVDGPData(NamedTuple):
    """Generated IV data."""
    Y: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    V_true: np.ndarray
    params: IVDGPParams
    true_semi_elasticity: float


def generate_iv_data_heteroskedastic(
    n: int = 2000,
    beta: float = 0.5,
    rho: float = 0.8,
    gamma_z: float = 1.0,
    mu_z: float = 1.5,
    sigma_v: float = 1.0,
    sigma_eps: float = 0.6,
    a_hetero: float = 0.4,
    seed: int = 42
) -> IVDGPData:
    """
    Generate IV data with X-dependent heteroskedasticity.

    DGP:
        Z ~ N(mu_z, 1)
        V ~ N(0, sigma_v^2)
        X = gamma_z * Z + V
        eps ~ N(0, sigma_eps^2 * (1 + a_hetero * X^2))
        log Y = beta * X + rho * V + eps

    Under this DGP:
        True semi-elasticity = beta + a_hetero * sigma_eps^2 * E[X]
                             = beta + a_hetero * sigma_eps^2 * gamma_z * mu_z
    """
    np.random.seed(seed)

    # Generate data
    Z = np.random.randn(n) + mu_z
    V = sigma_v * np.random.randn(n)
    X = gamma_z * Z + V

    # Heteroskedastic errors
    eps_std = sigma_eps * np.sqrt(1 + a_hetero * X**2)
    eps = eps_std * np.random.randn(n)

    # Outcome
    log_Y = beta * X + rho * V + eps
    Y = np.exp(log_Y)

    # True semi-elasticity
    E_X = gamma_z * mu_z
    correction = a_hetero * sigma_eps**2 * E_X
    true_semi_elasticity = beta + correction

    params = IVDGPParams(
        n=n, beta=beta, rho=rho, gamma_z=gamma_z, mu_z=mu_z,
        sigma_v=sigma_v, sigma_eps=sigma_eps, a_hetero=a_hetero, seed=seed
    )

    return IVDGPData(
        Y=Y,
        X=X.reshape(-1, 1),
        Z=Z.reshape(-1, 1),
        V_true=V,
        params=params,
        true_semi_elasticity=true_semi_elasticity
    )


def generate_iv_data_homoskedastic(
    n: int = 2000,
    beta: float = 0.5,
    rho: float = 0.8,
    gamma_z: float = 1.0,
    mu_z: float = 1.5,
    sigma_v: float = 1.0,
    sigma_eps: float = 0.6,
    seed: int = 42
) -> IVDGPData:
    """Generate IV data with homoskedastic errors (a_hetero=0)."""
    return generate_iv_data_heteroskedastic(
        n=n, beta=beta, rho=rho, gamma_z=gamma_z, mu_z=mu_z,
        sigma_v=sigma_v, sigma_eps=sigma_eps, a_hetero=0.0, seed=seed
    )


@pytest.fixture
def small_heteroskedastic_data() -> IVDGPData:
    """Small dataset for fast tests with heteroskedasticity."""
    return generate_iv_data_heteroskedastic(n=1000, seed=42)


@pytest.fixture
def medium_heteroskedastic_data() -> IVDGPData:
    """Medium dataset for standard tests."""
    return generate_iv_data_heteroskedastic(n=3000, seed=42)


@pytest.fixture
def large_heteroskedastic_data() -> IVDGPData:
    """Large dataset for asymptotic tests."""
    return generate_iv_data_heteroskedastic(n=8000, seed=42)


@pytest.fixture
def homoskedastic_data() -> IVDGPData:
    """Dataset with homoskedastic errors (semi-elasticity = beta)."""
    return generate_iv_data_homoskedastic(n=2000, seed=42)


@pytest.fixture
def weak_instrument_data() -> IVDGPData:
    """Dataset with weak instrument (low gamma_z)."""
    return generate_iv_data_heteroskedastic(
        n=3000, gamma_z=0.2, seed=42
    )


@pytest.fixture
def strong_instrument_data() -> IVDGPData:
    """Dataset with strong instrument."""
    # Use moderate gamma_z to avoid numerical overflow from large X values
    return generate_iv_data_heteroskedastic(
        n=3000, gamma_z=1.5, mu_z=1.0, a_hetero=0.2, seed=42
    )


# Default neural network parameters for fast testing
DEFAULT_FAST_PARAMS = {
    'first_stage_params': {
        'arch_params': {'hidden_layers': [32, 32]},
        'fit_params': {'epochs': 30, 'patience': 10, 'verbose': False}
    },
    'm_params': {
        'arch_params': {'hidden_layers': [64, 64]},
        'fit_params': {'epochs': 50, 'patience': 15, 'verbose': False}
    },
    'density_params': {
        'arch_params': {'shared': {'hidden_layers': [64, 64]}},
        'fit_params': {'epochs': 50, 'patience': 15, 'verbose': False}
    },
    'omega_params': {
        'arch_params': {'hidden_layers': [32, 32]},
        'fit_params': {'epochs': 30, 'n_permutations': 3, 'verbose': False}
    },
    'lambda_params': {
        'arch_params': {'hidden_layers': [16, 16]},
        'fit_params': {'epochs': 20, 'verbose': False}
    },
    'n_mc_samples': 200
}


@pytest.fixture
def fast_fit_params() -> Dict:
    """Fast neural network parameters for quick tests."""
    return DEFAULT_FAST_PARAMS.copy()
