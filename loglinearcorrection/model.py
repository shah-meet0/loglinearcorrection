from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import KFold
import statsmodels.api as sm

from .utils import _apply_fixed_effects, _detect_variable_types
from .nonparametric import NNModelNuisance, NNModelDensity
from .results import DREEMR
#from .density import DensityModel, DensityModelResults


class DoublyRobustElasticityEstimatorModel:
    """
    Doubly robust estimator for elasticity estimation with fixed effects.

    This model prepares data for doubly robust elasticity estimation by detecting
    variable types, applying fixed effects transformations, and storing both
    original and demeaned data for subsequent estimation procedures.
    """

    def __init__(
        self,
        endog: npt.ArrayLike,
        exog: npt.ArrayLike,
        weights: npt.ArrayLike | None = None,
        fixed_effects: list[str] | list[int] | None = None,
        interest: list[str] | list[int] | None = None,
        ordinal: list[str] | list[int] | None = None,
        **kwargs
    ) -> None:
        """
        Initialize the doubly robust elasticity estimator.

        Parameters
        ----------
        endog : array_like
            Dependent variable (outcome). Must contain only numeric data.
        exog : array_like
            Independent variables (regressors). Must contain only numeric data.
            Can be a pandas DataFrame, Series, or numpy array.
        weights : array_like, optional
            Observation weights. Must be numeric if provided.
        fixed_effects : list of str or list of int, optional
            Variable names (if `exog` is DataFrame) or column indices indicating
            which variables are fixed effects. These will be used for demeaning
            but not for variable type detection.
        interest : list of str or list of int, optional
            Variable names (if `exog` is DataFrame) or column indices indicating
            variables of primary interest.
        ordinal : list of str or list of int, optional
            Variable names (if `exog` is DataFrame) or column indices indicating
            which variables should be treated as ordinal. Variables not specified
            as ordinal will be automatically classified as either binary (if they
            have exactly 2 unique values) or continuous (otherwise). Note that
            variables with exactly 2 unique values will remain classified as binary
            even if specified in the ordinal parameter, as binary is a special case
            that requires specific handling.
        **kwargs
            Additional keyword arguments (reserved for future use).

        Attributes
        ----------
        endog : ndarray
            Original dependent variable as 1-D array.
        exog : ndarray
            Original independent variables as 2-D array.
        weights : ndarray or None
            Observation weights.
        endog_names : str
            Name of dependent variable (from input or default "y").
        exog_names : list of str
            Names of independent variables (from input or default "x1", "x2", ...).
        endog_demeaned : ndarray
            Dependent variable after fixed effects transformation.
        exog_demeaned : ndarray
            Independent variables after fixed effects transformation.
        variable_types : dict
            Mapping of variable indices/names to detected types ('continuous',
            'binary', 'ordinal') for non-fixed-effect variables. Binary variables
            are automatically detected (exactly 2 unique values), ordinal must be
            explicitly specified via the `ordinal` parameter, and all others are
            treated as continuous.
        fixed_effects : list or None
            Fixed effects specification.
        interest : list or None
            Variables of interest specification.
        ordinal : list or None
            Ordinal variables specification.
        nobs : int
            Number of observations.

        Raises
        ------
        ValueError
            If endog, exog, or weights contain non-numeric data.

        Notes
        -----
        The initialization performs the following operations:

        1. Extracts and stores variable names from pandas objects if provided,
           or generates default names ("y" for endog, "x1", "x2", ... for exog)
        2. Converts all inputs to numpy arrays and validates numeric types
        3. Detects variable types for non-fixed-effect variables:
           - Binary: Automatically detected when exactly 2 unique values (takes
             precedence over ordinal specification)
           - Ordinal: Must be explicitly specified via `ordinal` parameter
           - Continuous: All other variables
        4. Applies within-group demeaning for fixed effects using
           :func:`_apply_fixed_effects`
        5. Stores both original and transformed data for estimation
        
        Variable type detection is important for proper handling in elasticity
        estimation. Users should carefully specify ordinal variables based on
        their domain knowledge, as automatic detection can be unreliable. Binary
        detection takes precedence to ensure proper handling of dichotomous
        variables.
        """
        # Store specifications
        self.fixed_effects = fixed_effects
        self.interest = interest
        self.ordinal = ordinal

        # Extract names from pandas objects or generate defaults
        self.endog_names = (
            endog.name if isinstance(endog, pd.Series)
            else endog.columns[0] if isinstance(endog, pd.DataFrame)
            else "y"
        )
        
        # Convert to array first to get shape for default naming
        exog_arr = np.asarray(exog)
        n_vars = exog_arr.shape[1] if exog_arr.ndim == 2 else 1
        
        self.exog_names = (
            exog.columns.tolist() if isinstance(exog, pd.DataFrame)
            else [exog.name] if isinstance(exog, pd.Series)
            else [f"x{i+1}" for i in range(n_vars)]
        )

        # Convert to arrays and validate numeric types
        endog_arr = np.asarray(endog)
        weights_arr = np.asarray(weights) if weights is not None else None

        if not np.issubdtype(endog_arr.dtype, np.number):
            raise ValueError("endog must contain only numeric data")
        if not np.issubdtype(exog_arr.dtype, np.number):
            raise ValueError("exog must contain only numeric data")
        if weights_arr is not None and not np.issubdtype(weights_arr.dtype, np.number):
            raise ValueError("weights must contain only numeric data")

        # Store original data
        self.endog = endog_arr.ravel()
        self.exog = exog_arr if exog_arr.ndim == 2 else exog_arr.reshape(-1, 1)
        self.weights = weights_arr
        self.nobs = len(self.endog)

        # Identify non-fixed-effect variable indices
        if fixed_effects is None:
            non_fe_indices = list(range(self.exog.shape[1]))
        else:
            if self.exog_names and isinstance(fixed_effects[0], str):
                fe_indices = {self.exog_names.index(name) for name in fixed_effects}
            else:
                fe_indices = set(fixed_effects)
            non_fe_indices = [i for i in range(self.exog.shape[1]) if i not in fe_indices]

        # Detect variable types for non-fixed-effect variables
        self.variable_types = _detect_variable_types(
            self.exog[:, non_fe_indices], non_fe_indices
        )
        
        # Override with explicit ordinal specification (but respect binary detection)
        if ordinal is not None:
            if self.exog_names and isinstance(ordinal[0], str):
                ordinal_indices = [self.exog_names.index(name) for name in ordinal]
            else:
                ordinal_indices = list(ordinal)
            
            for idx in ordinal_indices:
                if idx in self.variable_types:  # Only if not a fixed effect
                    # Only override to ordinal if not already detected as binary
                    if self.variable_types[idx] != "binary":
                        self.variable_types[idx] = "ordinal"

        # Apply fixed effects transformation
        self.endog_demeaned, self.exog_demeaned = _apply_fixed_effects(
            self.endog, self.exog, fixed_effects, self.exog_names
        )


    def fit(self, n_folds: int = 5, random_state: Optional[int] = None,
            m_params: Dict = None, density_params:Dict=None,) -> 'DREEMR':
        """
        Fit the Doubly Robust Nonparametric Orthogonal (DR.NO) elasticity estimator.
        
        Implements cross-fitted estimation of elasticities using Neyman-orthogonalized 
        moment conditions for both continuous (semi-elasticities) and discrete 
        (percentage changes) variables.
        
        Parameters
        ----------
        n_folds : int, default=10
            Number of folds for cross-fitting.
        random_state : int or None, default=None
            Random state for reproducible fold splitting.
        fit_params : dict or None, default=None
            Additional parameters to pass to NPModel and DensityModel fit methods.
            
        Returns
        -------
        DREEMR
            DoublyRobustElasticityEstimatorModelResults object containing:
            - Elasticity estimates for variables of interest
            - Standard errors (to be implemented later)
            - First-stage OLS coefficients
            - Diagnostics from cross-fitting including NPModelResults and 
              DensityModelResults objects for each fold
            
        Notes
        -----
        The estimator follows these steps for each fold:
        1. Split data into training and evaluation sets
        2. Estimate OLS coefficients β on training data
        3. Estimate nuisance functions (m(x), f(x)) on training data  
        4. Compute influence functions α(x) on evaluation data
        5. Construct orthogonalized moments ψ
        6. Average moments across folds for final estimates
        """
        m_params = self._parse_m_params(m_params)
        density_params = self._parse_density_params(density_params)

        # Determine which variables to compute elasticities for
        if self.interest is None:
            # All non-fixed-effect variables
            if self.fixed_effects is None:
                interest_indices = list(range(self.exog.shape[1]))
            else:
                if self.exog_names and isinstance(self.fixed_effects[0], str):
                    fe_indices = {self.exog_names.index(name) for name in self.fixed_effects}
                else:
                    fe_indices = set(self.fixed_effects)
                interest_indices = [i for i in range(self.exog.shape[1]) if i not in fe_indices]
        else:
            # Convert interest specification to indices
            if self.exog_names and isinstance(self.interest[0], str):
                interest_indices = [self.exog_names.index(name) for name in self.interest]
            else:
                interest_indices = list(self.interest)
        
        # Initialize storage for cross-fitted moments
        n_params = len(interest_indices) + self.exog.shape[1]  # elasticities + β coefficients
        moments = np.zeros((self.nobs, n_params))
        fold_weights = np.zeros(self.nobs)  # Track which observations were in test sets
        
        # Set up cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Storage for diagnostics
        ols_coefficients = []
        m_results_folds = []  # NPModelResults objects for each fold
        f_results_folds = []  # DensityModelResults objects for each fold
        alpha_x_folds = []  # α(x) arrays for each fold
        theta_x_folds = []  # θ(x) arrays for each fold
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.exog)):
            # Split data
            endog_train, endog_test = self.endog_demeaned[train_idx], self.endog_demeaned[test_idx]
            exog_train, exog_test = self.exog_demeaned[train_idx], self.exog_demeaned[test_idx]
            
            # Original data for nuisance functions (includes fixed effects)
            # Need to remove fe from here
            exog_orig_train, exog_orig_test = self.exog[train_idx], self.exog[test_idx]
            endog_orig_train, endog_orig_test = self.endog[train_idx], self.endog[test_idx]
            
            weights_train = self.weights[train_idx] if self.weights is not None else None
            weights_test = self.weights[test_idx] if self.weights is not None else None
            
            # Step 1: Estimate OLS coefficients β on training data
            if weights_train is not None:
                ols_model = sm.WLS(np.log(endog_orig_train), exog_train, weights=weights_train)
            else:
                ols_model = sm.OLS(np.log(endog_orig_train), exog_train)
            
            ols_results = ols_model.fit()
            beta = ols_results.params
            ols_coefficients.append(beta)
            
            # Compute residuals on test data
            log_residuals_test = np.log(endog_orig_test) - exog_test @ beta
            exp_residuals_test = np.exp(log_residuals_test)  # exp(u_i)
            
            # Step 2: Estimate nuisance functions on training data
            # Using original (non-demeaned) data for nuisance functions
            
            # Estimate m(x) = E[exp(u)|x] using NPModel
            log_residuals_train = np.log(endog_orig_train) - exog_train @ beta
            exp_residuals_train = np.exp(log_residuals_train)


            # Create NPModel with variable types
            print('Training Nuisance Model for fold', fold_idx+1)
            m_model = NNModelNuisance(variable_types=self.variable_types, **m_params['arch_params'])
            m_results = m_model.fit(exog_orig_train, exp_residuals_train, **m_params['fit_params'])
            m_results_folds.append(m_results)
            
            # Predict m(x) on test data
            # m_prime_test = m'(x) for continuous interest variables, mhat = m(x) for all interest variables, mgrad = m(x+delta) for discrete interest variables
            m_test, m_prime_test = m_results.derivative(exog_orig_test, interest_indices)


            
            # Estimate density f(x) using DensityModel
            print('Training Density Model for fold', fold_idx+1)
            density_model = NNModelDensity(variable_types=self.variable_types, **density_params['arch_params'])
            f_results = density_model.fit(exog_orig_train, interest=interest_indices, **density_params['fit_params'])
            f_results_folds.append(f_results)

            # alpha_weight = f'(x)/f(x) for continuous interest variables, no idea for binary yet
            alpha_weights = f_results.alpha_weight(exog_orig_test, interest_indices)
            
            # Step 3: Construct moment conditions for each variable of interest
            fold_alpha_x = []  # α(x) for this fold
            fold_theta_x = []  # θ(x) for this fold






            
            for moment_idx, var_idx in enumerate(interest_indices):
                # i is where we'll get results for that var_idx
                var_type = self.variable_types.get(var_idx, 'continuous')
                
                if var_type == 'continuous':
                    # Continuous variable: semi-elasticity
                    # ε = β_k + m_k(x)/m(x)
                    # α(x) = -f_k(x)/(f(x)*m(x))
                    
                    # Get semi-elasticity from density model
                    alpha_weight = alpha_weights[:, moment_idx]

                    # Influence function
                    alpha_test = -alpha_weight / (m_test + 1e-10)
                    fold_alpha_x.append(alpha_test)
                    
                    # Orthogonalized moment (without elasticity parameter - that's what we solve for)
                    # g = β_k + m_k(x)/m(x) and φ = α(x)*p where p = exp(u) - m(x)
                    p_test = exp_residuals_test - m_test
                    
                    # Get m_k(x)/m(x) from NPModelResults
                    m_semi_elast_test = m_prime_test[:, moment_idx]/(m_test + 1e-10)
                    
                    # θ(x) = β_k + m_k/m
                    theta_test = beta[var_idx] + m_semi_elast_test
                    fold_theta_x.append(theta_test)
                    
                    # Store moment: g + φ = β_k + m_k/m + α*p
                    moments[test_idx, moment_idx] = theta_test + alpha_test * p_test
                    
                elif var_type == 'binary':
                    # Binary variable: percentage change with Δ=1
                    # First term (density derivative) is zero for binaries
                    # ε = exp(β*Δ) * m(x+Δ)/m(x) - 1
                    # α(x) = exp(β*Δ) * [0 - m(x+Δ)/m^2(x)]  (first term zero for binary)
                    
                    # delta = 1
                    # beta_delta = beta[var_idx] * delta
                    #
                    # # Create shifted x
                    # x_shifted = exog_orig_test.copy()
                    # x_shifted[:, var_idx] = 1 - x_shifted[:, var_idx]  # Flip binary variable
                    #
                    # # Predict m(x+Δ)
                    # m_shifted_test = m_results.predict(x_shifted)
                    #
                    # # Influence function (first term zero for binary)
                    # alpha_test = np.exp(beta_delta) * (-m_shifted_test / (m_test**2 + 1e-10))
                    # fold_alpha_x.append(alpha_test)
                    #
                    # # Orthogonalized moment
                    # p_test = exp_residuals_test - m_test
                    #
                    # # θ(x) = exp(β*Δ) * m(x+Δ)/m(x) - 1
                    # ratio = m_shifted_test / (m_test + 1e-10)
                    # theta_test = np.exp(beta_delta) * ratio - 1
                    # fold_theta_x.append(theta_test)
                    
                    # Store moment: g + φ

                    theta_test = 0
                    alpha_test = 0
                    p_test = 0
                    moments[test_idx, moment_idx] = theta_test + alpha_test * p_test
                    
                elif var_type == 'ordinal':
                    # # Ordinal variable: percentage change with Δ=1
                    # # ε = exp(β*Δ) * m(x+Δ)/m(x) - 1
                    # # α(x) = exp(β*Δ) * [f(x-Δ)/(m(x-Δ)*f(x)) - m(x+Δ)/m^2(x)]
                    #
                    # delta = 1
                    # beta_delta = beta[var_idx] * delta
                    #
                    # # Create shifted x values
                    # x_plus = exog_orig_test.copy()
                    # x_plus[:, var_idx] += delta
                    #
                    # x_minus = exog_orig_test.copy()
                    # x_minus[:, var_idx] -= delta
                    #
                    # # Predict m and f at shifted points
                    # m_plus_test = m_results.predict(x_plus)
                    # m_minus_test = m_results.predict(x_minus)
                    # f_minus_test = f_results.predict(x_minus)
                    #
                    # # Influence function
                    # term1 = f_minus_test / (m_minus_test * f_test + 1e-10)
                    # term2 = m_plus_test / (m_test**2 + 1e-10)
                    # alpha_test = np.exp(beta_delta) * (term1 - term2)
                    # fold_alpha_x.append(alpha_test)
                    #
                    # # Orthogonalized moment
                    # p_test = exp_residuals_test - m_test
                    #
                    # # θ(x) = exp(β*Δ) * m(x+Δ)/m(x) - 1
                    # ratio = m_plus_test / (m_test + 1e-10)
                    # theta_test = np.exp(beta_delta) * ratio - 1
                    # fold_theta_x.append(theta_test)
                    #
                    # Store moment
                    theta_test = 0
                    alpha_test = 0
                    p_test = 0
                    moments[test_idx, moment_idx] = theta_test + alpha_test * p_test

            
            # Store fold diagnostics
            alpha_x_folds.append(np.column_stack(fold_alpha_x) if fold_alpha_x else np.array([]))
            theta_x_folds.append(np.column_stack(fold_theta_x) if fold_theta_x else np.array([]))
            
            # Add OLS moment conditions (these don't need orthogonalization)
            for j in range(self.exog.shape[1]):
                moments[test_idx, len(interest_indices) + j] = (
                    log_residuals_test * exog_test[:, j]
                )
            
            # Track fold weights for averaging
            if weights_test is not None:
                fold_weights[test_idx] = weights_test
            else:
                fold_weights[test_idx] = 1.0
        
        # Step 4: Compute final estimates via method of moments
        # # Weight by observation weights if provided # Double weighting?
        # weighted_moments = moments * fold_weights[:, np.newaxis]
        moment_means = np.average(moments, axis=0, weights=fold_weights)

        w = fold_weights.reshape(-1, 1)  # (n,1)
        W = w / w.sum()  # normalize
        V = moments.T @ (W * moments)  # (k,k)
        
        # Extract elasticity estimates (first len(interest_indices) moments)
        elasticity_estimates = moment_means[:len(interest_indices)]
        elasticity_variances = np.diag(V)[:len(interest_indices)]
        
        # Create results dictionary
        elasticity_dict = {}
        for i, var_idx in enumerate(interest_indices):
            var_name = self.exog_names[var_idx] if self.exog_names else f"x{var_idx+1}"
            elasticity_dict[var_name] = {
                'estimate': elasticity_estimates[i],
                'std_est': np.sqrt(elasticity_variances[i]/self.nobs),  # Placeholder for standard error
                'type': self.variable_types.get(var_idx, 'continuous'),
                'index': var_idx
            }
        
        # Average OLS coefficients across folds
        avg_beta = np.mean(ols_coefficients, axis=0)
        
        # Create and return results object
        results = DREEMR(
            elasticities=elasticity_dict,
            beta=avg_beta,
            exog_names=self.exog_names,
            endog_name=self.endog_names,
            n_folds=n_folds,
            nobs=self.nobs,
            variable_types=self.variable_types,
            fold_diagnostics={
                'ols_coefficients': ols_coefficients,
                'moment_means': moment_means,
                'm_results': m_results_folds,  # NPModelResults objects
                'f_results': f_results_folds,  # DensityModelResults objects
                'alpha_x': alpha_x_folds,  # α(x) arrays
                'theta_x': theta_x_folds   # θ(x) arrays
            }
        )
        
        return results

    def _parse_m_params(self, m_params):
        if m_params is None:
            m_params = {'arch_params': {'hidden_layers': [512, 512, 512],
                                        'input_size': 0,
                                        'output_size': 0},
                        'fit_params': {}}

        for key in m_params.keys():
            if key not in ['arch_params', 'fit_params']:
                raise ValueError(f"Invalid key in m_params: {key} . Must be 'arch_params' or 'fit_params'.")

        if 'fit_params' not in m_params:
            m_params['fit_params'] = {}

        if 'arch_params' not in m_params:
            m_params['arch_params'] = {'hidden_layers': [256, 256, 256],
                                       'input_size': 0,
                                       'output_size': 0}

        m_params['arch_params']['input_size'] = self.exog_demeaned.shape[1]
        m_params['arch_params']['output_size'] = 1

        return m_params

    def _parse_density_params(self, density_params):
        if density_params is None:
            density_params = {'arch_params': {}, 'fit_params': {}}

            density_params['arch_params'] = {
                'shared': {},
                'score': {},
                'cond': {}
            }

            density_params['fit_params'] = {}

        for key in density_params.keys():
            if key not in ['arch_params', 'fit_params']:
                raise ValueError(f"Invalid key in density_params: {key} . Must be 'arch_params' or 'fit_params'.")

        if 'arch_params' not in density_params:
            density_params['arch_params'] = {
                'shared': {},
                'score': {},
                'cond': {}
            }

        if 'fit_params' not in density_params:
            density_params['fit_params'] = {}

        return density_params


DREEM = DoublyRobustElasticityEstimatorModel
