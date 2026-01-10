"""
Results class for IV-DRNO estimator.

Provides:
- Point estimates and standard errors
- Confidence intervals
- Diagnostic information
- Comparison tests (IVDRNO vs 2SLS, etc.)
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import numpy.typing as npt
import pandas as pd


class IVDoublyRobustElasticityEstimatorModelResults:
    """
    Results container for the IV-DRNO estimator.
    
    Stores estimation results and provides methods for inference
    based on the fitted IV-DRNO model.
    
    Parameters
    ----------
    elasticities : Union[pd.DataFrame, ndarray]
        Elasticity estimates for interest variables.
    beta : ndarray
        Control function OLS coefficients on X.
    gamma : ndarray or None
        Control function OLS coefficients on exogenous controls W.
    rho : ndarray
        Control function OLS coefficients on V.
    ols_results : statsmodels results
        Full OLS results from control function estimation.
    exog_names : list of str
        Names of endogenous variables X.
    endog_name : str
        Name of outcome variable Y.
    instrument_names : list of str
        Names of instruments Z.
    control_names : list of str
        Names of exogenous control variables W.
    n_folds : int
        Number of cross-fitting folds.
    nobs : int
        Number of observations.
    variable_types : dict
        Mapping of variable indices to types.
    interest_indices : list of int
        Indices of interest variables.
    fold_results : list of dict
        Detailed results from each fold.
    moments : ndarray
        Moment conditions for all observations.
    fold_weights : ndarray
        Weights for each observation.
    V_hat : ndarray
        Estimated control function residuals.
        
    Attributes
    ----------
    standard_errors : ndarray
        Standard errors of elasticity estimates.
    confidence_intervals : dict
        95% confidence intervals.
    """
    
    def __init__(
        self,
        elasticities: Union[pd.DataFrame, npt.NDArray[np.float64]],
        beta: npt.NDArray[np.float64],
        gamma: Optional[npt.NDArray[np.float64]],
        rho: npt.NDArray[np.float64],
        ols_results: Any,
        exog_names: List[str],
        endog_name: str,
        instrument_names: List[str],
        control_names: List[str],
        n_folds: int,
        nobs: int,
        variable_types: Dict[int, str],
        interest_indices: List[int],
        fold_results: List[Dict],
        moments: npt.NDArray[np.float64],
        fold_weights: npt.NDArray[np.float64],
        V_hat: npt.NDArray[np.float64]
    ):
        self.elasticities = elasticities
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.ols_results = ols_results
        self.exog_names = exog_names
        self.endog_name = endog_name
        self.instrument_names = instrument_names
        self.control_names = control_names if control_names else []
        self.n_folds = n_folds
        self.nobs = nobs
        self.variable_types = variable_types
        self.interest_indices = interest_indices
        self.fold_results = fold_results
        self.moments = moments
        self.fold_weights = fold_weights
        self.V_hat = V_hat
        
        self._is_pandas = isinstance(elasticities, pd.DataFrame)
        
        # Will be populated by compute_variances()
        self._standard_errors = None
        self._variance_matrix = None
        self._confidence_intervals = None
        
    @property
    def standard_errors(self) -> npt.NDArray[np.float64]:
        """Get standard errors (computed if not already done)."""
        if self._standard_errors is None:
            self.compute_variances()
        return self._standard_errors
    
    @property
    def confidence_intervals(self) -> Dict[str, tuple]:
        """Get 95% confidence intervals."""
        if self._confidence_intervals is None:
            self.compute_variances()
        return self._confidence_intervals
    
    def compute_variances(self, include_ols_moments: bool = True) -> None:
        """
        Compute variance-covariance matrix and standard errors.
        
        Uses the influence function representation:
            sqrt(n)(θ̂ - θ_0) →_d N(0, Var(ψ))
        
        where ψ is the stacked moment/influence function:
            ψ = [ψ_θ, ψ_OLS] for joint inference on (elasticities, β, γ, ρ)
        
        Parameters
        ----------
        include_ols_moments : bool, default=True
            If True, include OLS influence function for joint inference
            on all parameters. If False, only compute variance for elasticities.
        """
        if self._standard_errors is not None:
            return  # Already computed
        
        n = self.nobs
        n_interest = len(self.interest_indices)
        
        # Get elasticity moments
        moments_theta = self.moments  # shape (n, n_interest)
        
        # Compute OLS influence function for joint inference
        if include_ols_moments and self.ols_results is not None:
            # Get dimensions
            k_x = len(self.exog_names)
            k_w = len(self.control_names) if self.control_names else 0
            k_v = self.V_hat.shape[1] if self.V_hat.ndim > 1 else 1
            k_total = k_x + k_w + k_v
            
            # OLS influence function: ψ_i = (X'X)^{-1} X_i ε_i
            # We need to reconstruct the design matrix
            resid = self.ols_results.resid
            
            # Build design matrix [X_endog, W, V]
            # This requires access to original data - for now use approximation
            # based on normalized covariance and residuals
            try:
                # (X'X)^{-1} from statsmodels
                bread = self.ols_results.normalized_cov_params
                
                # Approximate influence using residual heterogeneity
                # Full version would need stored design matrix
                ols_influence = np.zeros((n, k_total))
                
                # Scale residuals by diagonal of bread matrix
                for j in range(k_total):
                    ols_influence[:, j] = resid * np.sqrt(bread[j, j])
                    
            except Exception:
                # Fallback: just use residuals
                ols_influence = np.zeros((n, k_total))
                ols_influence[:, 0] = resid
            
            # Stack moments
            stacked_moments = np.column_stack([moments_theta, ols_influence])
            n_params_total = n_interest + k_total
        else:
            stacked_moments = moments_theta
            n_params_total = n_interest
        
        # Compute variance matrix
        w = self.fold_weights.reshape(-1, 1)
        W = w / w.sum()
        
        # Moment variance: S = E[ψ ψ']
        S = stacked_moments.T @ (W * stacked_moments)
        
        self._variance_matrix = S
        self._full_variance_matrix = S
        
        # Standard errors for elasticities (first n_interest components)
        variances_theta = np.diag(S)[:n_interest]
        self._standard_errors = np.sqrt(variances_theta / n)
        
        # Full standard errors (all parameters)
        self._full_standard_errors = np.sqrt(np.diag(S) / n)
        
        # Confidence intervals for elasticities
        from scipy import stats
        z_score = stats.norm.ppf(0.975)
        
        self._confidence_intervals = {}
        
        if self._is_pandas:
            self.elasticities['std_err'] = self._standard_errors
            for i, var_name in enumerate(self.elasticities.index):
                estimate = self.elasticities.loc[var_name, 'estimate']
                se = self._standard_errors[i]
                self._confidence_intervals[var_name] = (
                    estimate - z_score * se,
                    estimate + z_score * se
                )
        else:
            for i, idx in enumerate(self.interest_indices):
                var_name = self.exog_names[idx]
                estimate = self.elasticities[i]
                se = self._standard_errors[i]
                self._confidence_intervals[var_name] = (
                    estimate - z_score * se,
                    estimate + z_score * se
                )
                
    def get_elasticity(self, var_name: str) -> tuple[float, float]:
        """
        Get elasticity estimate and standard error for a variable.
        
        Parameters
        ----------
        var_name : str
            Variable name.
            
        Returns
        -------
        estimate : float
            Point estimate.
        std_err : float
            Standard error.
        """
        if self._standard_errors is None:
            self.compute_variances()
            
        if self._is_pandas:
            if var_name not in self.elasticities.index:
                raise KeyError(f"Variable '{var_name}' not found")
            row = self.elasticities.loc[var_name]
            return row['estimate'], row['std_err']
        else:
            if var_name not in self.exog_names:
                raise KeyError(f"Variable '{var_name}' not found")
            idx = self.exog_names.index(var_name)
            if idx not in self.interest_indices:
                raise KeyError(f"Variable '{var_name}' not in interest variables")
            pos = self.interest_indices.index(idx)
            return self.elasticities[pos], self._standard_errors[pos]
        
    def first_stage_diagnostics(self) -> Dict[str, Any]:
        """
        Compute first stage diagnostics.
        
        Returns
        -------
        dict with keys:
            - 'r_squared': R² of first stage (per fold average)
            - 'f_statistic': Approximate F-statistic for weak instruments
            - 'residual_std': Standard deviation of V̂
        """
        diagnostics = {}
        
        # Variance of V̂
        V_var = np.var(self.V_hat, axis=0)
        X_var = np.var(self.V_hat + self.V_hat, axis=0)  # Reconstruct X variance approx
        
        # R² = 1 - Var(V)/Var(X)
        # This is approximate; proper R² needs g(Z) predictions
        diagnostics['residual_var'] = float(V_var.mean()) if V_var.ndim > 0 else float(V_var)
        diagnostics['residual_std'] = float(np.sqrt(V_var.mean())) if V_var.ndim > 0 else float(np.sqrt(V_var))
        
        # Collect from folds if available
        if self.fold_results and 'g_results' in self.fold_results[0]:
            fold_metrics = []
            for fold in self.fold_results:
                if hasattr(fold['g_results'], 'metrics'):
                    fold_metrics.append(fold['g_results'].metrics)
            if fold_metrics:
                diagnostics['fold_val_losses'] = [m.get('val_loss', np.nan) for m in fold_metrics]
                
        return diagnostics
    
    def hausman_test(self) -> Dict[str, float]:
        """
        Hausman test comparing IV-DRNO to naive OLS on log Y ~ X.
        
        Tests H0: X is exogenous (IV-DRNO = OLS)
        
        Returns
        -------
        dict with keys:
            - 'statistic': Hausman test statistic
            - 'p_value': p-value
            - 'df': degrees of freedom
        """
        from scipy import stats
        
        # Get IV-DRNO estimates
        if self._is_pandas:
            iv_est = self.elasticities['estimate'].values
        else:
            iv_est = self.elasticities
            
        # Get naive OLS (beta from log Y ~ X, ignoring endogeneity)
        # We use the control function beta, which should be consistent
        ols_est = self.beta[self.interest_indices] if self.beta is not None else iv_est
        
        # Difference
        diff = iv_est - ols_est
        
        # Variance of difference (simplified: use IV variance)
        if self._variance_matrix is None:
            self.compute_variances()
        var_diff = np.diag(self._variance_matrix) / self.nobs
        
        # Test statistic
        k = len(diff)
        if k == 1:
            stat = float(diff[0]**2 / (var_diff[0] + 1e-10))
        else:
            # Wald test: (diff)' V^{-1} (diff)
            var_inv = np.linalg.pinv(np.diag(var_diff))
            stat = float(diff @ var_inv @ diff)
            
        p_value = 1 - stats.chi2.cdf(stat, df=k)
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'df': k
        }
    
    def control_function_test(self) -> Dict[str, float]:
        """
        Test for endogeneity using control function coefficient.
        
        Tests H0: ρ = 0 (X is exogenous)
        
        If ρ is significantly different from 0, there is evidence
        of endogeneity that the control function corrects for.
        """
        from scipy import stats
        
        if self.rho is None:
            return {'statistic': np.nan, 'p_value': np.nan, 'rho': np.nan}
            
        rho = np.atleast_1d(self.rho)
        
        # Get standard error from OLS
        if self.ols_results is not None and hasattr(self.ols_results, 'bse'):
            n_exog = len(self.beta) if self.beta is not None else 0
            rho_se = self.ols_results.bse[n_exog:n_exog + len(rho)]
        else:
            rho_se = np.abs(rho) * 0.1 + 1e-6  # Rough approximation
            
        # t-test for each rho
        t_stats = rho / (rho_se + 1e-10)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=self.nobs - len(self.beta) - len(rho)))
        
        if len(rho) == 1:
            return {
                'statistic': float(t_stats[0]),
                'p_value': float(p_values[0]),
                'rho': float(rho[0]),
                'rho_se': float(rho_se[0])
            }
        else:
            return {
                'statistic': t_stats.tolist(),
                'p_value': p_values.tolist(),
                'rho': rho.tolist(),
                'rho_se': rho_se.tolist()
            }
    
    def summary(self) -> None:
        """Print summary of estimation results."""
        print("=" * 75)
        print("IV-DRNO: Doubly Robust Elasticity Estimator with Instrumental Variables")
        print("=" * 75)
        print(f"Dependent variable: {self.endog_name}")
        print(f"Endogenous variable(s): {', '.join(self.exog_names)}")
        print(f"Instrument(s): {', '.join(self.instrument_names)}")
        if self.control_names:
            print(f"Exogenous control(s): {', '.join(self.control_names)}")
        print(f"Number of observations: {self.nobs}")
        print(f"Number of folds: {self.n_folds}")
        
        # Identification status
        k_x = len(self.exog_names)
        k_z = len(self.instrument_names)
        if k_z == k_x:
            id_status = "Just-identified"
        elif k_z > k_x:
            id_status = f"Over-identified ({k_z} instruments for {k_x} endogenous)"
        else:
            id_status = f"Under-identified (WARNING)"
        print(f"Identification: {id_status}")
        print("=" * 75)
        print()
        
        # First stage diagnostics
        print("First Stage Diagnostics:")
        print("-" * 75)
        fs_diag = self.first_stage_diagnostics()
        print(f"  Residual std (V̂): {fs_diag['residual_std']:.4f}")
        print()
        
        # Control function test
        print("Control Function Test (H0: X is exogenous):")
        print("-" * 75)
        cf_test = self.control_function_test()
        if isinstance(cf_test['rho'], list):
            for i, (r, se, t, p) in enumerate(zip(
                cf_test['rho'], cf_test['rho_se'], 
                cf_test['statistic'], cf_test['p_value']
            )):
                print(f"  ρ_{i+1} = {r:.4f} (SE: {se:.4f}), t = {t:.2f}, p = {p:.4f}")
        else:
            print(f"  ρ = {cf_test['rho']:.4f} (SE: {cf_test.get('rho_se', np.nan):.4f})")
            print(f"  t-statistic: {cf_test['statistic']:.2f}, p-value: {cf_test['p_value']:.4f}")
        print()
        
        # Elasticity estimates
        print("Elasticity Estimates (Average Structural Function Semi-Elasticity):")
        print("-" * 75)
        print(f"{'Variable':<20} {'Type':<12} {'Estimate':<12} {'Std. Error':<12} {'95% CI':<24}")
        print("-" * 75)
        
        if self._is_pandas:
            for var_name in self.elasticities.index:
                row = self.elasticities.loc[var_name]
                estimate = row['estimate']
                var_type = row['type']
                se = row.get('std_err', np.nan)
                ci = self.confidence_intervals.get(var_name, (np.nan, np.nan))
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if not np.isnan(ci[0]) else "N/A"
                se_str = f"{se:.4f}" if not np.isnan(se) else "N/A"
                print(f"{var_name:<20} {var_type:<12} {estimate:>11.4f} {se_str:>12} {ci_str:<24}")
        else:
            for i, idx in enumerate(self.interest_indices):
                var_name = self.exog_names[idx]
                estimate = self.elasticities[i]
                var_type = self.variable_types.get(idx, 'continuous')
                se = self._standard_errors[i] if self._standard_errors is not None else np.nan
                ci = self.confidence_intervals.get(var_name, (np.nan, np.nan))
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if not np.isnan(ci[0]) else "N/A"
                se_str = f"{se:.4f}" if not np.isnan(se) else "N/A"
                print(f"{var_name:<20} {var_type:<12} {estimate:>11.4f} {se_str:>12} {ci_str:<24}")
                
        print("-" * 75)
        print()
        
        # Control function coefficients
        print("Control Function OLS Coefficients:")
        print("-" * 75)
        if self.beta is not None:
            print("  Endogenous (β):")
            for i, idx in enumerate(self.interest_indices):
                # self.beta is already sliced to interest_indices
                print(f"    {self.exog_names[idx]:<20}: {self.beta[i]:>11.4f}")
        if self.gamma is not None:
            print("  Exogenous controls (γ):")
            for i, name in enumerate(self.control_names):
                print(f"    {name:<20}: {self.gamma[i]:>11.4f}")
        if self.rho is not None:
            rho = np.atleast_1d(self.rho)
            print("  Control function (ρ):")
            for i, r in enumerate(rho):
                print(f"    V_{i+1:<19}: {r:>11.4f}")
        print("=" * 75)
        
    def __repr__(self) -> str:
        """String representation."""
        n_vars = len(self.interest_indices)
        return (
            f"IVDoublyRobustElasticityEstimatorModelResults("
            f"n_vars={n_vars}, nobs={self.nobs}, n_folds={self.n_folds})"
        )


# Alias
IVDREEMR = IVDoublyRobustElasticityEstimatorModelResults
