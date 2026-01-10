"""
Example usage and tests for the IV-DRNO estimator.

This script demonstrates:
1. Data generation with endogeneity
2. Fitting the IV-DRNO estimator
3. Comparing to naive OLS and 2SLS
4. Handling multiple endogenous variables
5. Including exogenous controls
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def generate_iv_data(
    n: int = 2000,
    beta_true: float = 0.5,
    rho_true: float = 0.8,
    gamma_true: float = 1.0,
    sigma_v: float = 1.0,
    sigma_eps: float = 0.5,
    seed: int = 42
) -> tuple:
    """
    Generate data with endogeneity for IV-DRNO testing (scalar X case).
    
    Model:
        X = gamma * Z + V
        log Y = beta * X + rho * V + epsilon
        
    Where V ~ N(0, sigma_v^2) and epsilon ~ N(0, sigma_eps^2).
    
    The endogeneity arises because V affects both X and Y.
    Z is a valid instrument: correlated with X, excludable from Y.
    """
    np.random.seed(seed)
    
    # Instrument
    Z = np.random.randn(n)
    
    # Confounder/control function
    V = sigma_v * np.random.randn(n)
    
    # Endogenous treatment (nonlinear first stage for realism)
    X = gamma_true * Z + 0.3 * Z**2 + V
    
    # Error in outcome
    epsilon = sigma_eps * np.random.randn(n)
    
    # Outcome (log-linear)
    log_Y = beta_true * X + rho_true * V + epsilon
    Y = np.exp(log_Y)
    
    return Y, X.reshape(-1, 1), Z.reshape(-1, 1), V


def generate_iv_data_multiple(
    n: int = 2000,
    k_x: int = 2,
    k_z: int = 3,
    seed: int = 42
) -> tuple:
    """
    Generate data with multiple endogenous variables and instruments.
    
    Model:
        X = Γ Z + V,  where X is k_x dim, Z is k_z dim, V is k_x dim
        log Y = β'X + ρ'V + ε
        
    Over-identified if k_z > k_x.
    """
    np.random.seed(seed)
    
    # True parameters
    beta_true = np.array([0.5, -0.3])[:k_x]
    rho_true = np.array([0.8, 0.6])[:k_x]
    
    # Instruments
    Z = np.random.randn(n, k_z)
    
    # First stage coefficients (k_x x k_z)
    Gamma = np.random.randn(k_x, k_z) * 0.5
    Gamma[np.arange(min(k_x, k_z)), np.arange(min(k_x, k_z))] = 1.0  # Strong diagonal
    
    # Confounder
    V = np.random.randn(n, k_x)
    
    # Endogenous treatment
    X = Z @ Gamma.T + V
    
    # Error
    epsilon = 0.5 * np.random.randn(n)
    
    # Outcome
    log_Y = X @ beta_true + V @ rho_true + epsilon
    Y = np.exp(log_Y)
    
    return Y, X, Z, V, beta_true, rho_true


def generate_iv_data_with_controls(
    n: int = 2000,
    k_w: int = 2,
    seed: int = 42
) -> tuple:
    """
    Generate data with endogeneity and exogenous controls.
    
    Model:
        X = g(Z, W) + V
        log Y = β X + γ'W + ρ V + ε
    """
    np.random.seed(seed)
    
    beta_true = 0.5
    rho_true = 0.8
    gamma_true = np.array([0.3, -0.2])[:k_w]
    
    # Instruments and controls
    Z = np.random.randn(n, 1)
    W = np.random.randn(n, k_w)
    
    # Confounder
    V = np.random.randn(n)
    
    # Endogenous treatment (depends on Z and W)
    X = Z.ravel() + 0.5 * W[:, 0] + V
    
    # Outcome
    epsilon = 0.5 * np.random.randn(n)
    log_Y = beta_true * X + W @ gamma_true + rho_true * V + epsilon
    Y = np.exp(log_Y)
    
    return Y, X.reshape(-1, 1), Z, W, V, beta_true, gamma_true, rho_true


def test_ivdrno_basic():
    """Basic functionality test with scalar X."""
    print("=" * 60)
    print("Test 1: Basic IV-DRNO estimation (scalar X)")
    print("=" * 60)
    
    # Generate data
    Y, X, Z, V_true = generate_iv_data(n=1000, seed=123)
    beta_true = 0.5
    
    # Naive OLS (biased)
    X_with_const = sm.add_constant(X)
    ols_naive = sm.OLS(np.log(Y), X_with_const).fit()
    print(f"\nNaive OLS coefficient: {ols_naive.params[1]:.4f}")
    print(f"True beta: {beta_true:.4f}")
    print(f"OLS bias: {ols_naive.params[1] - beta_true:.4f}")
    
    # 2SLS for comparison
    Z_with_const = sm.add_constant(Z)
    first_stage = sm.OLS(X.ravel(), Z_with_const).fit()
    X_hat = first_stage.fittedvalues.reshape(-1, 1)
    X_hat_with_const = sm.add_constant(X_hat)
    tsls = sm.OLS(np.log(Y), X_hat_with_const).fit()
    print(f"\n2SLS coefficient: {tsls.params[1]:.4f}")
    print(f"2SLS bias: {tsls.params[1] - beta_true:.4f}")
    
    # Control function OLS
    V_hat = X.ravel() - X_hat.ravel()
    cf_design = np.column_stack([X, V_hat.reshape(-1, 1)])
    cf_ols = sm.OLS(np.log(Y), cf_design).fit()
    print(f"\nControl Function OLS:")
    print(f"  beta (X coef): {cf_ols.params[0]:.4f}")
    print(f"  rho (V coef): {cf_ols.params[1]:.4f}")
    
    print("\n" + "=" * 60)


def test_multiple_endogenous():
    """Test with multiple endogenous variables (over-identified)."""
    print("=" * 60)
    print("Test 2: Multiple endogenous variables (over-identified)")
    print("=" * 60)
    
    # Generate data with 2 endogenous vars and 3 instruments
    Y, X, Z, V_true, beta_true, rho_true = generate_iv_data_multiple(
        n=1000, k_x=2, k_z=3, seed=456
    )
    
    print(f"\nTrue beta: {beta_true}")
    print(f"True rho: {rho_true}")
    print(f"Number of instruments: {Z.shape[1]} (over-identified)")
    
    # Naive OLS
    ols_naive = sm.OLS(np.log(Y), X).fit()
    print(f"\nNaive OLS coefficients: {ols_naive.params}")
    print(f"OLS bias: {ols_naive.params - beta_true}")
    
    print("\n" + "=" * 60)


def test_with_controls():
    """Test with exogenous control variables."""
    print("=" * 60)
    print("Test 3: With exogenous control variables")
    print("=" * 60)
    
    Y, X, Z, W, V_true, beta_true, gamma_true, rho_true = generate_iv_data_with_controls(
        n=1000, k_w=2, seed=789
    )
    
    print(f"\nTrue beta: {beta_true}")
    print(f"True gamma (controls): {gamma_true}")
    print(f"True rho: {rho_true}")
    
    # Naive OLS with controls
    design = np.column_stack([X, W])
    ols_naive = sm.OLS(np.log(Y), design).fit()
    print(f"\nNaive OLS coefficients:")
    print(f"  X coef: {ols_naive.params[0]:.4f} (true: {beta_true})")
    print(f"  W coefs: {ols_naive.params[1:]} (true: {gamma_true})")
    
    print("\n" + "=" * 60)


def test_density_ratio():
    """Test the density ratio estimation."""
    print("\n" + "=" * 60)
    print("Test 4: Density ratio estimation")
    print("=" * 60)
    
    try:
        from loglinearcorrection.iv_nonparametric import NNModelDensityRatio
    except ImportError:
        from iv_nonparametric import NNModelDensityRatio
    
    np.random.seed(42)
    n = 1000
    
    # Generate (X, V) with known dependence
    V = np.random.randn(n)
    X = 0.5 * V + np.random.randn(n)  # X depends on V
    
    # Reshape
    X = X.reshape(-1, 1)
    V = V.reshape(-1, 1)
    
    # Fit density ratio model
    omega_model = NNModelDensityRatio(hidden_layers=[64, 64])
    omega_results = omega_model.fit(
        X, V, 
        epochs=50, 
        n_permutations=3,
        verbose=True
    )
    
    # Predict omega
    omega_pred = omega_results.predict(X, V)
    
    print(f"\nOmega statistics:")
    print(f"  Mean: {omega_pred.mean():.4f}")
    print(f"  Std:  {omega_pred.std():.4f}")
    print(f"  Min:  {omega_pred.min():.4f}")
    print(f"  Max:  {omega_pred.max():.4f}")
    

def test_riesz_lambda():
    """Test the Riesz lambda estimation via autodiff."""
    print("\n" + "=" * 60)
    print("Test 5: Riesz lambda (automatic DML) estimation")
    print("=" * 60)
    
    try:
        from loglinearcorrection.iv_nonparametric import NNModelRieszLambda
    except ImportError:
        from iv_nonparametric import NNModelRieszLambda
    
    np.random.seed(42)
    n = 1000
    
    # Generate Z and a "pathwise derivative" that depends on Z
    Z = np.random.randn(n, 2)
    true_lambda = 0.5 * Z[:, 0] + 0.3 * Z[:, 1]
    noise = 0.1 * np.random.randn(n)
    pathwise_deriv = true_lambda + noise
    
    # Fit lambda model
    lambda_model = NNModelRieszLambda(hidden_layers=[32, 32])
    lambda_results = lambda_model.fit(
        Z, pathwise_deriv,
        epochs=100,
        verbose=True
    )
    
    # Predict lambda
    lambda_pred = lambda_results.predict(Z)
    
    # Compare to true
    mse = np.mean((lambda_pred - true_lambda)**2)
    corr = np.corrcoef(lambda_pred, true_lambda)[0, 1]
    
    print(f"\nLambda estimation:")
    print(f"  MSE vs true lambda: {mse:.6f}")
    print(f"  Correlation with true: {corr:.4f}")


def run_all_tests():
    """Run all tests."""
    test_ivdrno_basic()
    test_multiple_endogenous()
    test_with_controls()
    
    # These require PyTorch
    try:
        import torch
        test_density_ratio()
        test_riesz_lambda()
    except ImportError as e:
        print(f"\nSkipping PyTorch-dependent tests: {e}")


if __name__ == "__main__":
    run_all_tests()
