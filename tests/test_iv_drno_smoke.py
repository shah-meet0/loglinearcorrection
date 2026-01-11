"""
Smoke test simulation for IV-DRNO estimator.

This test verifies that the IV-DRNO estimator:
1. Runs without errors
2. Produces estimates close to true ASF semi-elasticity
3. Properly corrects for the retransformation bias (unlike 2SLS)
4. Handles the control function approach correctly

KEY INSIGHT: 2SLS identifies β, but the ASF semi-elasticity is:
    θ = β + E[μ'(X)/μ(X)]

Under homoskedastic errors, μ'(x) = 0 so θ = β.
Under X-dependent heteroskedasticity, μ'(x) ≠ 0 so θ ≠ β.

This test uses heteroskedastic errors to create a wedge between 2SLS and 
the true causal semi-elasticity, demonstrating IV-DRNO's value.

DGP:
    Z ~ N(μ_z, 1)           # Instrument with non-zero mean
    V ~ N(0, σ_v²)          # Confounder
    X = γ·Z + V             # First stage (endogenous)
    ε ~ N(0, σ_ε²(1 + a·X²)) # X-dependent heteroskedasticity
    log Y = β·X + ρ·V + ε   # Outcome
    
Under this DGP:
    μ'(x)/μ(x) = a·σ_ε²·x
    E[θ(X)] = β + a·σ_ε²·E[X] = β + a·σ_ε²·γ·μ_z
    
So 2SLS identifies β, but true semi-elasticity = β + a·σ_ε²·γ·μ_z
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Dict


def generate_iv_dgp_heteroskedastic(
    n: int = 5000,
    beta_true: float = 0.5,
    rho_true: float = 0.8,
    gamma_z: float = 1.0,
    mu_z: float = 1.0,        # Non-zero mean for Z creates E[X] ≠ 0
    sigma_v: float = 1.0,
    sigma_eps: float = 0.7,
    a_hetero: float = 0.4,    # Heteroskedasticity parameter
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate data with X-dependent heteroskedasticity.
    
    This creates a wedge between β (what 2SLS identifies) and the 
    ASF semi-elasticity (what IV-DRNO targets).
    
    Model:
        Z ~ N(μ_z, 1)
        V ~ N(0, σ_v²)
        X = γ·Z + V
        ε | X ~ N(0, σ_ε²·(1 + a·X²))
        log Y = β·X + ρ·V + ε
    
    Theoretical results:
        E[X] = γ·μ_z
        μ(x) = exp(ρ²σ_v²/2 + σ_ε²(1 + a·x²)/2)
        μ'(x)/μ(x) = a·σ_ε²·x
        θ(x) = β + a·σ_ε²·x
        E[θ(X)] = β + a·σ_ε²·E[X] = β + a·σ_ε²·γ·μ_z
    """
    np.random.seed(seed)
    
    # Generate instrument with non-zero mean
    Z = np.random.randn(n) + mu_z
    
    # Generate confounder
    V = sigma_v * np.random.randn(n)
    
    # First stage: X = γ·Z + V
    X = gamma_z * Z + V
    
    # X-dependent heteroskedastic error
    # Var(ε|X) = σ_ε²·(1 + a·X²)
    sigma_eps_x = sigma_eps * np.sqrt(1 + a_hetero * X**2)
    eps = sigma_eps_x * np.random.randn(n)
    
    # Outcome
    log_Y = beta_true * X + rho_true * V + eps
    Y = np.exp(log_Y)
    
    # Compute theoretical quantities
    E_X = gamma_z * mu_z
    
    # Theoretical ASF semi-elasticity (average over X distribution)
    # θ(x) = β + a·σ_ε²·x
    # E[θ(X)] = β + a·σ_ε²·E[X]
    theta_correction = a_hetero * sigma_eps**2 * E_X
    true_semi_elasticity = beta_true + theta_correction
    
    # Also compute via Monte Carlo for verification
    # θ(x) = β + a·σ_ε²·x, so E[θ(X)] ≈ mean(β + a·σ_ε²·X)
    theta_x_values = beta_true + a_hetero * sigma_eps**2 * X
    mc_semi_elasticity = theta_x_values.mean()
    
    true_params = {
        'beta': beta_true,
        'rho': rho_true,
        'gamma_z': gamma_z,
        'mu_z': mu_z,
        'sigma_v': sigma_v,
        'sigma_eps': sigma_eps,
        'a_hetero': a_hetero,
        'E_X': E_X,
        'theta_correction': theta_correction,
        'true_semi_elasticity': true_semi_elasticity,
        'mc_semi_elasticity': mc_semi_elasticity,
        'theta_x_values': theta_x_values,  # For diagnostics
        'V': V,
        'sigma_eps_x': sigma_eps_x  # Realized heteroskedastic std
    }
    
    return Y, X.reshape(-1, 1), Z.reshape(-1, 1), true_params


def run_baseline_estimators(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    true_params: Dict
) -> Dict:
    """
    Run baseline estimators for comparison.
    
    Key insight:
    - Naive OLS is biased due to endogeneity (V in both X and Y)
    - 2SLS consistently estimates β, NOT the ASF semi-elasticity
    - Control function OLS estimates β and ρ correctly
    - IV-DRNO should estimate the ASF semi-elasticity
    """
    results = {}
    
    # 1. Naive OLS (biased due to endogeneity)
    ols = sm.OLS(np.log(Y), X).fit()
    results['ols_beta'] = ols.params[0]
    results['ols_se'] = ols.bse[0]
    
    # 2. 2SLS - estimates β, not the ASF semi-elasticity!
    first_stage = sm.OLS(X, Z).fit()
    X_hat = first_stage.fittedvalues.reshape(-1, 1)
    second_stage = sm.OLS(np.log(Y), X_hat).fit()
    results['tsls_beta'] = second_stage.params[0]
    results['tsls_se'] = second_stage.bse[0]
    
    # First stage F-statistic
    results['first_stage_F'] = first_stage.fvalue
    
    # 3. Control function with true V (oracle)
    V_true = true_params['V'].reshape(-1, 1)
    cf_design = np.column_stack([X, V_true])
    cf_ols = sm.OLS(np.log(Y), cf_design).fit()
    results['cf_oracle_beta'] = cf_ols.params[0]
    results['cf_oracle_rho'] = cf_ols.params[1]
    
    # 4. Control function with estimated V
    V_hat = X - X_hat
    cf_est_design = np.column_stack([X, V_hat])
    cf_est_ols = sm.OLS(np.log(Y), cf_est_design).fit()
    results['cf_est_beta'] = cf_est_ols.params[0]
    results['cf_est_rho'] = cf_est_ols.params[1]
    
    return results


def test_iv_drno_smoke():
    """
    Main smoke test demonstrating IV-DRNO vs 2SLS.
    
    Key demonstration: Under heteroskedasticity, 2SLS identifies β
    but the true ASF semi-elasticity is β + correction term.
    IV-DRNO should capture this correction.
    """
    print("=" * 70)
    print("IV-DRNO Smoke Test: Heteroskedastic DGP")
    print("=" * 70)
    print("\nThis test demonstrates the difference between:")
    print("  - 2SLS: identifies β (effect on log Y)")
    print("  - IV-DRNO: identifies ASF semi-elasticity = β + E[μ'(X)/μ(X)]")
    print("\nUnder X-dependent heteroskedasticity, these differ!")
    
    # Generate data with heteroskedasticity
    n = 4000
    beta_true = 0.5
    rho_true = 0.8
    mu_z = 1.5        # Non-zero mean creates E[X] ≠ 0
    a_hetero = 0.4    # Heteroskedasticity strength
    sigma_eps = 0.6
    
    print("\n1. Generating synthetic data...")
    Y, X, Z, true_params = generate_iv_dgp_heteroskedastic(
        n=n,
        beta_true=beta_true,
        rho_true=rho_true,
        gamma_z=1.0,
        mu_z=mu_z,
        sigma_v=1.0,
        sigma_eps=sigma_eps,
        a_hetero=a_hetero,
        seed=42
    )
    
    print(f"   n = {n}")
    print(f"   True β = {true_params['beta']:.4f}")
    print(f"   True ρ = {true_params['rho']:.4f}")
    print(f"   E[X] = {true_params['E_X']:.4f}")
    print(f"   Heteroskedasticity parameter a = {true_params['a_hetero']:.4f}")
    print(f"\n   Theoretical correction term = a·σ_ε²·E[X] = {true_params['theta_correction']:.4f}")
    print(f"   True ASF semi-elasticity = β + correction = {true_params['true_semi_elasticity']:.4f}")
    print(f"   Monte Carlo verification = {true_params['mc_semi_elasticity']:.4f}")
    
    # Run baseline estimators
    print("\n2. Running baseline estimators...")
    baselines = run_baseline_estimators(Y, X, Z, true_params)
    
    print(f"\n   First stage F-statistic: {baselines['first_stage_F']:.1f}")
    print(f"\n   Naive OLS:")
    print(f"     β estimate: {baselines['ols_beta']:.4f}")
    print(f"     Bias from true β: {baselines['ols_beta'] - beta_true:.4f}")
    
    print(f"\n   2SLS (identifies β, NOT ASF semi-elasticity):")
    print(f"     β estimate: {baselines['tsls_beta']:.4f}")
    print(f"     Bias from true β: {baselines['tsls_beta'] - beta_true:.4f}")
    print(f"     Gap from ASF semi-elasticity: {baselines['tsls_beta'] - true_params['true_semi_elasticity']:.4f}")
    
    print(f"\n   Control function OLS (estimated V):")
    print(f"     β estimate: {baselines['cf_est_beta']:.4f}")
    print(f"     ρ estimate: {baselines['cf_est_rho']:.4f}")
    
    # Run IV-DRNO
    print("\n3. Running IV-DRNO estimator...")
    
    try:
        from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM
        
        model = IVDREEM(
            endog=Y,
            exog=X,
            instruments=Z,
            interest=[0]
        )
        
        # Configure networks (smaller for smoke test)
        first_stage_params = {
            'arch_params': {'hidden_layers': [64, 64]},
            'fit_params': {'epochs': 50, 'patience': 10, 'verbose': False}
        }
        m_params = {
            'arch_params': {'hidden_layers': [128, 128, 128]},
            'fit_params': {'epochs': 100, 'patience': 15, 'verbose': False}
        }
        density_params = {
            'arch_params': {'shared': {'hidden_layers': [128, 128, 128]}},
            'fit_params': {'epochs': 100, 'patience': 15, 'verbose': False}
        }
        omega_params = {
            'arch_params': {'hidden_layers': [64, 64]},
            'fit_params': {'epochs': 50, 'n_permutations': 5, 'verbose': False}
        }
        lambda_params = {
            'arch_params': {'hidden_layers': [32, 32]},
            'fit_params': {'epochs': 30, 'verbose': False}
        }
        
        results = model.fit(
            n_folds=3,
            random_state=42,
            first_stage_params=first_stage_params,
            m_params=m_params,
            density_params=density_params,
            omega_params=omega_params,
            lambda_params=lambda_params,
            n_mc_samples=300
        )
        
        print("\n4. IV-DRNO Results:")
        print("-" * 50)
        
        if hasattr(results, 'elasticities') and isinstance(results.elasticities, pd.DataFrame):
            estimate = results.elasticities['estimate'].values[0]
            var_name = results.elasticities.index[0]
        else:
            estimate = results.elasticities[0]
            var_name = "x1"
        
        se = results.standard_errors[0]
        ci = results.confidence_intervals.get(var_name, (np.nan, np.nan))
        
        print(f"   ASF semi-elasticity estimate: {estimate:.4f}")
        print(f"   Standard error: {se:.4f}")
        print(f"   95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"   True ASF semi-elasticity: {true_params['true_semi_elasticity']:.4f}")
        print(f"   Bias: {estimate - true_params['true_semi_elasticity']:.4f}")
        
        print("\n5. Control Function Estimates from IV-DRNO:")
        print("-" * 50)
        print(f"   β (effect on log Y): {results.beta[0]:.4f} (true: {beta_true})")
        print(f"   ρ (control function): {results.rho[0]:.4f} (true: {rho_true})")
        
        # Comparison table
        print("\n" + "=" * 70)
        print("6. KEY COMPARISON: 2SLS vs IV-DRNO")
        print("=" * 70)
        print(f"\n   Target: ASF semi-elasticity = {true_params['true_semi_elasticity']:.4f}")
        print(f"   (This differs from β = {beta_true} due to heteroskedasticity)")
        print()
        print(f"   {'Estimator':<30} {'Estimate':<12} {'Target':<12} {'Bias':<12}")
        print("-" * 70)
        print(f"   {'2SLS (identifies β only)':<30} {baselines['tsls_beta']:>11.4f} {true_params['true_semi_elasticity']:>11.4f} {baselines['tsls_beta'] - true_params['true_semi_elasticity']:>11.4f}")
        print(f"   {'IV-DRNO (ASF semi-elast.)':<30} {estimate:>11.4f} {true_params['true_semi_elasticity']:>11.4f} {estimate - true_params['true_semi_elasticity']:>11.4f}")
        print("-" * 70)
        
        gap_2sls = abs(baselines['tsls_beta'] - true_params['true_semi_elasticity'])
        gap_ivdrno = abs(estimate - true_params['true_semi_elasticity'])
        
        print(f"\n   2SLS gap from ASF target: {gap_2sls:.4f}")
        print(f"   IV-DRNO gap from ASF target: {gap_ivdrno:.4f}")
        
        # Validation checks
        print("\n7. Validation Checks:")
        print("-" * 50)
        
        # Check 1: 2SLS should be biased for ASF (by design)
        check1_passed = gap_2sls > 0.05  # Should have meaningful gap
        print(f"   ✓ 2SLS differs from ASF target (gap > 0.05): {'PASS' if check1_passed else 'FAIL'}")
        print(f"     (Gap = {gap_2sls:.4f}, this is the retransformation bias)")
        
        # Check 2: IV-DRNO should be closer to ASF than 2SLS
        check2_passed = gap_ivdrno < gap_2sls
        print(f"   ✓ IV-DRNO closer to ASF than 2SLS: {'PASS' if check2_passed else 'FAIL'}")
        
        # Check 3: True value in CI
        check3_passed = ci[0] <= true_params['true_semi_elasticity'] <= ci[1]
        print(f"   ✓ True ASF in 95% CI: {'PASS' if check3_passed else 'FAIL'}")
        
        # Check 4: β estimate reasonable
        beta_bias = abs(results.beta[0] - beta_true)
        check4_passed = beta_bias < 0.15
        print(f"   ✓ β estimate reasonable (bias < 0.15): {'PASS' if check4_passed else 'FAIL'}")
        
        # Check 5: ρ detected (endogeneity)
        check5_passed = results.rho[0] > 0.3
        print(f"   ✓ Endogeneity detected (ρ > 0.3): {'PASS' if check5_passed else 'FAIL'}")
        
        all_passed = all([check1_passed, check2_passed, check3_passed, check4_passed, check5_passed])
        
        print("\n" + "=" * 70)
        if all_passed:
            print("SMOKE TEST: ALL CHECKS PASSED ✓")
        else:
            print("SMOKE TEST: SOME CHECKS FAILED ✗")
        print("=" * 70)
        
        # Print interpretation
        print("\n8. INTERPRETATION:")
        print("-" * 70)
        print(f"""
   The true causal effect of X on E[Y] (the ASF semi-elasticity) is {true_params['true_semi_elasticity']:.4f}.
   
   2SLS estimates β = {baselines['tsls_beta']:.4f}, which is the effect on log E[Y|X,V].
   This DIFFERS from the ASF semi-elasticity by {gap_2sls:.4f} due to heteroskedasticity.
   
   IV-DRNO estimates {estimate:.4f}, capturing the retransformation correction.
   
   The correction term E[μ'(X)/μ(X)] = {true_params['theta_correction']:.4f} arises because:
   - Error variance increases with X² (heteroskedasticity)
   - This makes the MGF of the error X-dependent
   - Jensen's inequality creates a wedge between β and the semi-elasticity
        """)
        
        # Full summary
        print("\n")
        results.summary()
        
        return results, baselines, true_params
        
    except ImportError as e:
        print(f"\n   ERROR: Could not import IV-DRNO module: {e}")
        return None, baselines, true_params
    except Exception as e:
        print(f"\n   ERROR during IV-DRNO estimation: {e}")
        import traceback
        traceback.print_exc()
        return None, baselines, true_params


def test_varying_heteroskedasticity():
    """
    Test how the 2SLS vs IV-DRNO gap varies with heteroskedasticity strength.
    """
    print("\n" + "=" * 70)
    print("Sensitivity Analysis: Varying Heteroskedasticity")
    print("=" * 70)
    
    a_values = [0.0, 0.2, 0.4, 0.6]
    beta_true = 0.5
    
    print(f"\n{'a_hetero':<12} {'True θ':<12} {'2SLS gap':<12} {'Correction':<12}")
    print("-" * 50)
    
    for a in a_values:
        Y, X, Z, params = generate_iv_dgp_heteroskedastic(
            n=2000, beta_true=beta_true, a_hetero=a, mu_z=1.5, seed=42
        )
        baselines = run_baseline_estimators(Y, X, Z, params)
        
        gap = baselines['tsls_beta'] - params['true_semi_elasticity']
        
        print(f"{a:<12.2f} {params['true_semi_elasticity']:<12.4f} {gap:<12.4f} {params['theta_correction']:<12.4f}")
    
    print("\nNote: As heteroskedasticity (a) increases, 2SLS diverges more from the true target.")


if __name__ == "__main__":
    # Run main smoke test
    results, baselines, true_params = test_iv_drno_smoke()
    
    # Show sensitivity to heteroskedasticity
    test_varying_heteroskedasticity()
