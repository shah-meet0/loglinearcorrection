"""
Results classes for the doubly robust elasticity estimator.
"""
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import pandas as pd

class DoublyRobustElasticityEstimatorModelResults:
    """
    Results container for the Doubly Robust Elasticity Estimator.
    
    This class stores estimation results and provides methods for prediction
    and inference based on the fitted DR.NO model.
    
    Parameters
    ----------
    elasticities : Union[pd.DataFrame, npt.ndarray]
        Dictionary mapping variable names to elasticity information:
        - 'estimate': float - Point estimate of elasticity
        - 'type': str - Variable type ('continuous', 'binary', 'ordinal')
        - 'index': int - Original column index in exog
    beta : ndarray
        Averaged OLS coefficients from cross-fitting, shape (k_vars,)
    exog_names : list of str
        Names of exogenous variables
    endog_name : str
        Name of endogenous variable
    n_folds : int
        Number of folds used in cross-fitting
    nobs : int
        Number of observations
    variable_types : dict
        Mapping of variable indices to types
    fold_diagnostics : dict
        Diagnostic information from cross-fitting:
        - 'ols_coefficients': list of ndarray - OLS coefficients per fold
        - 'moment_means': ndarray - Mean of moment conditions
        - 'm_results': list of NPModelResults - Nuisance function m(x) results
        - 'f_results': list of DensityModelResults - Density f(x) results
        - 'alpha_x': list of ndarray - Influence functions α(x)
        - 'theta_x': list of ndarray - Elasticity estimates θ(x)
    var_params : dict, optional
        Parameters for variance estimation:
        - 'method': str - Variance method ('influence', 'gmm', 'bootstrap')
        - Additional method-specific parameters
    
    Attributes
    ----------
    standard_errors : ndarray or None
        Standard errors of elasticity estimates (computed on first access)
    confidence_intervals : dict or None
        95% confidence intervals (computed on first access)
    
    Methods
    -------
    predict_semi_elasticity(x, var_index)
        Compute semi-elasticity at new points for a specific variable
    predict_log_y(x)
        Predict log(y) at new points
    predict_y(x)
        Predict y at new points
    summary()
        Print summary of results
    get_elasticity(var_name)
        Get elasticity estimate and standard error for a variable
    """

    def __init__(
        self,
        elasticities: Union[pd.DataFrame, npt.NDArray[np.float64]],
        beta: npt.NDArray[np.float64],
        ols_results: Any,
        exog_names: List[str],
        endog_name: str,
        n_folds: int,
        nobs: int,
        variable_types: Dict[int, str],
        interest_indices: List[int],
        fold_results: List[Dict],
        moments: npt.NDArray[np.float64],
        derivative: npt.NDArray[np.float64],
        fold_weights: npt.NDArray[np.float64],
        gamma: Optional[npt.NDArray[np.float64]] = None
    ):
        self.elasticities = elasticities
        self.beta = beta
        self.ols_results = ols_results
        self.exog_names = exog_names
        self.endog_name = endog_name
        self.n_folds = n_folds
        self.nobs = nobs
        self.variable_types = variable_types
        self.interest_indices = interest_indices
        self.fold_results = fold_results
        self.moments = moments
        self.derivative = derivative
        self.fold_weights = fold_weights
        self._ppml_fit = gamma is not None
        self.gamma = gamma
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
    
    def compute_variances(self) -> None:
        """Compute variance-covariance matrix and standard errors."""
        if self._standard_errors is not None:
            return  # Already computed
            
        w = self.fold_weights.reshape(-1, 1)
        W = w / w.sum()
        S = self.moments.T @ (W * self.moments)

        D = np.average(self.derivative, axis=0, weights=self.fold_weights)
        D_inv = np.linalg.pinv(D)
        V = D_inv @ S @ D_inv.T
        
        self._variance_matrix = V
        elasticity_variances = np.diag(V)
        self._standard_errors = np.sqrt(elasticity_variances / self.nobs) # we might not want to use nobs, but sum of weights
        
        # Compute confidence intervals
        from scipy import stats
        z_score = stats.norm.ppf(0.975)
        
        self._confidence_intervals = {}
        if self._is_pandas:
            self.elasticities['std_err'] = self._standard_errors[[i for i in range(len(self.interest_indices))]]
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
        """Get elasticity estimate and standard error for a variable."""
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
    
    def predict_semi_elasticity(
        self, 
        x: npt.ArrayLike, 
        var_index: int,
        use_stored_theta: bool = False
    ) -> npt.NDArray[np.float64]:
        """
        Compute semi-elasticity at new points for a specific variable.
        
        The semi-elasticity θ(x) represents:
        - For continuous variables: β_k + m_k(x)/m(x)
        - For binary variables: exp(β·Δ) * m(x+Δ)/m(x) - 1  
        - For ordinal variables: exp(β·Δ) * m(x+Δ)/m(x) - 1
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate semi-elasticity, shape (n_samples, k_vars)
        var_index : int
            Index of variable for which to compute semi-elasticity
        use_stored_theta : bool, default=False
            If True and x matches original data points, use stored theta values
            from cross-fitting. Otherwise recompute using fitted nuisance functions.
        
        Returns
        -------
        ndarray
            Semi-elasticity values θ(x) at each point, shape (n_samples,)
        
        Notes
        -----
        This method computes the same θ(x) values that were computed during
        fitting. For the original data points, these are stored in 
        fold_diagnostics['theta_x']. For new points, we recompute using
        the fitted nuisance functions and average across folds.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Find which variable in the interest set this corresponds to
        if var_index not in self.interest_indices:
            raise ValueError(f"Variable at index {var_index} not in interest variables")
        interest_position = self.interest_indices.index(var_index)
        
        # If requested and available, use stored theta values
        if use_stored_theta and len(self.fold_results) > 0:
            theta_values = []
            for fold in self.fold_results:
                theta_fold = fold['theta_x']
                if theta_fold.size > 0 and interest_position < theta_fold.shape[1]:
                    # Note: This would need proper indexing to match x to test points
                    theta_values.append(theta_fold[:, interest_position])
            if theta_values:
                return np.mean(theta_values, axis=0)
        
        # Otherwise, compute theta(x) using the fitted nuisance functions
        var_type = self.variable_types.get(var_index, 'continuous')
        
        # Average theta computations across folds
        theta_predictions = []
        
        for fold in self.fold_results:
            m_results = fold['m_results']
            if var_type == 'continuous':
                # θ(x) = β_k + m_k(x)/m(x)
                m_x = m_results.predict(x)
                m_k_x = m_results.derivative(x, var_index) if hasattr(m_results, 'derivative') else None
                
                if m_k_x is not None:
                    theta = self.beta[var_index] + m_k_x / (m_x + 1e-10)
                else:
                    # Fallback to predict_semi_elasticity if derivative not available
                    semi_elast = m_results.predict_semi_elasticity(x, var_index)
                    theta = self.beta[var_index] + semi_elast
                    
            elif var_type == 'binary':
                # θ(x) = exp(β·Δ) * m(x+Δ)/m(x) - 1
                delta = 1
                beta_delta = self.beta[var_index] * delta
                
                m_x = m_results.predict(x)
                x_shifted = x.copy()
                x_shifted[:, var_index] = 1 - x_shifted[:, var_index]  # Flip binary
                m_shifted = m_results.predict(x_shifted)
                
                theta = np.exp(beta_delta) * (m_shifted / (m_x + 1e-10)) - 1
                
            elif var_type == 'ordinal':
                # θ(x) = exp(β·Δ) * m(x+Δ)/m(x) - 1
                delta = 1
                beta_delta = self.beta[var_index] * delta
                
                m_x = m_results.predict(x)
                x_shifted = x.copy()
                x_shifted[:, var_index] += delta
                m_shifted = m_results.predict(x_shifted)
                
                theta = np.exp(beta_delta) * (m_shifted / (m_x + 1e-10)) - 1
            else:
                raise ValueError(f"Unknown variable type: {var_type}")
            
            theta_predictions.append(theta)
        
        # Return average theta across folds
        return np.mean(theta_predictions, axis=0)
    
    def predict_log_y(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Predict log(y) at new points.
        
        Uses the averaged OLS coefficients: log(y) = β·x
        
        Parameters
        ----------
        x : array_like
            Predictor values, shape (n_samples, k_vars) or (k_vars,)
        
        Returns
        -------
        ndarray
            Predicted log(y) values, shape (n_samples,) or scalar
        """
        x = np.asarray(x)
        if x.ndim == 1:
            return np.dot(x, self.beta)
        return x @ self.beta
    
    def predict_y(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Predict y at new points.
        
        Uses the model: y = exp(β·x) * m(x) where m(x) = E[exp(u)|x]
        
        Parameters
        ----------
        x : array_like
            Predictor values, shape (n_samples, k_vars) or (k_vars,)
        
        Returns
        -------
        ndarray
            Predicted y values, shape (n_samples,) or scalar
        
        Notes
        -----
        This accounts for the retransformation bias by including the
        estimated conditional expectation of exp(u) given x.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        log_pred = self.predict_log_y(x)
        
        # Average m(x) predictions across folds
        m_predictions = []
        for fold in self.fold_results:
            m_predictions.append(fold['m_results'].predict(x))
        m_x = np.mean(m_predictions, axis=0)
        
        return np.exp(log_pred) * m_x
    

    def summary(self) -> None:
        """Print summary of estimation results."""
        print("=" * 70)
        print(f"Doubly Robust Elasticity Estimator Results")
        print(f"Dependent variable: {self.endog_name}")
        print(f"Number of observations: {self.nobs}")
        print(f"Number of folds: {self.n_folds}")
        print("=" * 70)
        print()
        print("Elasticity Estimates:")
        print("-" * 70)
        print(f"{'Variable':<20} {'Type':<12} {'Estimate':<12} {'Std. Error':<12} {'95% CI':<20}")
        print("-" * 70)
        
        if self._is_pandas:
            for var_name in self.elasticities.index:
                row = self.elasticities.loc[var_name]
                estimate = row['estimate']
                var_type = row['type']
                se = row.get('std_err', np.nan)
                
                ci = self.confidence_intervals.get(var_name, (np.nan, np.nan))
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if not np.isnan(ci[0]) else "N/A"
                se_str = f"{se:.4f}" if not np.isnan(se) else "N/A"
                
                print(f"{var_name:<20} {var_type:<12} {estimate:>11.4f} {se_str:>12} {ci_str:<20}")
        else:
            for i, idx in enumerate(self.interest_indices):
                var_name = self.exog_names[idx]
                estimate = self.elasticities[i]
                var_type = self.variable_types.get(idx, 'continuous')
                se = self._standard_errors[i] if self._standard_errors is not None else np.nan
                
                ci = self.confidence_intervals.get(var_name, (np.nan, np.nan))
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if not np.isnan(ci[0]) else "N/A"
                se_str = f"{se:.4f}" if not np.isnan(se) else "N/A"
                
                print(f"{var_name:<20} {var_type:<12} {estimate:>11.4f} {se_str:>12} {ci_str:<20}")
        
        print("-" * 70)
        print()
        print("OLS Coefficients:")
        print("-" * 70)
        for i, index in enumerate(self.interest_indices):
            print(f"{self.exog_names[index]:<20} Coef: {self.beta[i]:>11.4f}")
        print("-" * 70)
        if self._ppml_fit:
            print("-" * 70)
            print()
            print("PPML Coefficients:")
            print("-" * 70)
            for i, index in enumerate(self.interest_indices):
                print(f"{self.exog_names[index]:<20} Coef: {self.gamma[i]:>11.4f}")
            print("=" * 70)

    def __repr__(self) -> str:
        """String representation of results."""
        n_vars = len(self.interest_indices)
        return (
            f"DoublyRobustElasticityEstimatorModelResults("
            f"n_vars={n_vars}, nobs={self.nobs}, n_folds={self.n_folds})"
        )


# Alias for convenience
DREEMR = DoublyRobustElasticityEstimatorModelResults
