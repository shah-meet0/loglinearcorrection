"""
Results classes for the doubly robust elasticity estimator.
"""
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
import numpy as np
import numpy.typing as npt

class DoublyRobustElasticityEstimatorModelResults:
    """
    Results container for the Doubly Robust Elasticity Estimator.
    
    This class stores estimation results and provides methods for prediction
    and inference based on the fitted DR.NO model.
    
    Parameters
    ----------
    elasticities : dict
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
        elasticities: Dict[str, Dict[str, Any]],
        beta: npt.NDArray[np.float64],
        exog_names: List[str],
        endog_name: str,
        n_folds: int,
        nobs: int,
        variable_types: Dict[int, str],
        fold_diagnostics: Dict[str, Any],
        var_params: Optional[Dict[str, Any]] = None
    ):
        self.elasticities = elasticities
        self.beta = beta
        self.exog_names = exog_names
        self.endog_name = endog_name
        self.n_folds = n_folds
        self.nobs = nobs
        self.variable_types = variable_types
        self.fold_diagnostics = fold_diagnostics
        self.var_params = var_params or {}
        
        # Lazy computation of standard errors
        self._standard_errors = None
        self._confidence_intervals = None
        self._variance_results = None
    
    @property
    def standard_errors(self) -> npt.NDArray[np.float64]:
        """
        Compute standard errors using variance estimator.
        
        Lazily computed on first access using the variance estimation
        module and stored for subsequent use.
        
        Returns
        -------
        ndarray
            Standard errors for elasticity estimates
        """
        if self._standard_errors is None:
            self._compute_variance(self.var_params)
        return self._standard_errors
    
    @property
    def confidence_intervals(self) -> Dict[str, tuple]:
        """
        Compute 95% confidence intervals for elasticities.
        
        Returns
        -------
        dict
            Mapping from variable names to (lower, upper) confidence bounds
        """
        if self._confidence_intervals is None:
            self._compute_variance(self.var_params)
        return self._confidence_intervals
    
    def _compute_variance(self, var_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Compute variance estimates using VarianceModel.
        
        Uses the VarianceModel class to compute standard errors
        based on the cross-fitted influence functions stored in
        fold_diagnostics.
        
        Parameters
        ----------
        var_params : dict, optional
            Parameters to pass to VarianceModel constructor:
            - method: str, default='influence'
            - Additional method-specific parameters
        """
        try:
            from .variance import VarianceModel
            
            # Default variance parameters
            if var_params is None:
                var_params = {}
            
            # Create variance model
            var_model = VarianceModel(**var_params)
            
            # Fit variance model
            var_results = var_model.fit(
                fold_diagnostics=self.fold_diagnostics,
                n_folds=self.n_folds,
                nobs=self.nobs,
                elasticities=self.elasticities,
                weights=getattr(self, 'weights', None)
            )
            
            # Store results
            self._variance_results = var_results
            self._standard_errors = var_results.standard_errors
            
            # Extract confidence intervals
            self._confidence_intervals = {}
            for var_name in self.elasticities:
                ci_info = var_results.confidence_intervals.get(var_name, {})
                lower = ci_info.get('lower', np.nan)
                upper = ci_info.get('upper', np.nan)
                self._confidence_intervals[var_name] = (lower, upper)
                
        except ImportError:
            # Fallback if variance module not implemented
            import warnings
            warnings.warn(
                "variance.py module not found. Standard errors not available. "
                "Implement VarianceModel class for inference.",
                UserWarning
            )
            self._standard_errors = np.full(len(self.elasticities), np.nan)
            self._confidence_intervals = {
                name: (np.nan, np.nan) for name in self.elasticities
            }
    
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
        interest_vars = list(self.elasticities.keys())
        var_name = self.exog_names[var_index] if var_index < len(self.exog_names) else None
        
        if var_name not in interest_vars:
            raise ValueError(f"Variable at index {var_index} not in interest variables")
        
        # Get position in interest variables
        interest_position = interest_vars.index(var_name)
        
        # If requested and available, use stored theta values
        if use_stored_theta and len(self.fold_diagnostics.get('theta_x', [])) > 0:
            # Average stored theta values across folds for this variable
            theta_values = []
            for theta_fold in self.fold_diagnostics['theta_x']:
                if theta_fold.size > 0 and interest_position < theta_fold.shape[1]:
                    theta_values.append(theta_fold[:, interest_position])
            
            if theta_values:
                # Note: This assumes x corresponds to the original test points
                # In practice, would need to match/interpolate points
                return np.mean(theta_values, axis=0)
        
        # Otherwise, compute theta(x) using the fitted nuisance functions
        var_type = self.variable_types.get(var_index, 'continuous')
        
        # Average theta computations across folds
        theta_predictions = []
        
        for m_results in self.fold_diagnostics['m_results']:
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
        
        # Get log predictions
        log_pred = self.predict_log_y(x)
        
        # Average m(x) predictions across folds
        m_predictions = []
        for m_results in self.fold_diagnostics['m_results']:
            m_predictions.append(m_results.predict(x))
        m_x = np.mean(m_predictions, axis=0)
        
        # Return exp(β·x) * m(x)
        return np.exp(log_pred) * m_x
    
    @property
    def variance_results(self) -> Optional['VarianceModelResults']:
        """
        Get the full variance model results object.
        
        Returns
        -------
        VarianceModelResults or None
            Full variance results if computed, None otherwise
        """
        if self._variance_results is None and self._standard_errors is None:
            # Trigger computation if not yet done
            self._compute_variance(self.var_params)
        return self._variance_results

    
    def summary(self) -> None:
        """
        Print summary of estimation results.
        
        Displays elasticity estimates, standard errors, and confidence
        intervals in a formatted table.
        """
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
        
        for var_name, var_info in self.elasticities.items():
            estimate = var_info['estimate']
            var_type = var_info['type']
            
            # Get standard error and CI
            try:
                _, se = self.get_elasticity(var_name)
                ci = self.confidence_intervals.get(var_name, (np.nan, np.nan))
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if not np.isnan(ci[0]) else "N/A"
                se_str = f"{se:.4f}" if not np.isnan(se) else "N/A"
            except:
                se_str = "N/A"
                ci_str = "N/A"
            
            print(f"{var_name:<20} {var_type:<12} {estimate:>11.4f} {se_str:>12} {ci_str:<20}")
        
        print("-" * 70)
        print()
        print("OLS Coefficients (averaged across folds):")
        print("-" * 70)
        for i, name in enumerate(self.exog_names):
            print(f"{name:<20} {self.beta[i]:>11.4f}")
        print("=" * 70)
    
    def __repr__(self) -> str:
        """String representation of results."""
        n_vars = len(self.elasticities)
        return (
            f"DoublyRobustElasticityEstimatorModelResults("
            f"n_vars={n_vars}, nobs={self.nobs}, n_folds={self.n_folds})"
        )


# Alias for convenience
DREEMR = DoublyRobustElasticityEstimatorModelResults
