import numpy as np
import pandas as pd
from statsmodels.base.model import Model, Results
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import joblib

def _estimate_single_point_worker(args):
    """Worker function for parallel estimation (must be at module level for pickling)."""
    x_point, results_obj, variable_idx = args
    x_point = x_point.reshape(1, -1)
    return results_obj.estimate_at_point(x_point, variable_idx)


class DoublyRobustElasticityEstimator(Model):
    r"""Doubly robust estimator for elasticities and semi-elasticities in log-transformed models.
    
    This class implements a doubly robust estimator that corrects for the bias that occurs
    in log-transformed OLS models by estimating the conditional expectation of the
    exponentiated residuals and its gradient.
    
    Parameters
    ----------
    endog : array-like
        Response variable (untransformed)
    exog : array-like
        Regressor matrix
    interest : int, str, list, or None, optional
        Index or name of the regressor of interest, or list of indices/names, 
        or None to calculate for all variables. Defaults to None.
    log_x : bool or list, optional
        Whether the regressor(s) of interest is/are log-transformed. 
        Can be a single bool for all variables or a list matching interest. Defaults to False.
    estimator_type : str, optional
        Type of estimator to use for the nonparametric component: 
        'kernel' (local linear kernel regression), 'nn' (neural network), 'ols' (OLS), or 'binary'
        Defaults to 'kernel'
    elasticity : bool, optional
        Whether to estimate elasticities (True) or semi-elasticities (False), defaults to False
    kernel_params : dict, optional
        Parameters for the kernel regression estimator
    nn_params : dict, optional
        Parameters for the neural network estimator
    density_estimator : str, optional
        Type of density estimator to use for calculating asymptotic variance: 
        'kernel' (kernel density estimation) or 'nn' (neural network density estimation)
        Defaults to 'kernel'
    density_params : dict, optional
        Parameters for the density estimator
    fe : int or list, optional
        Indices of fixed effects variables, defaults to None
        
    Notes
    -----
    The estimator is based on the formula:
    \hat{z}(x) = \hat\beta + \frac{\hat{m}'(x; \hat\beta)}{\hat{m}(x; \hat\beta)}
    
    where \hat{m}(x) is a local linear kernel regression for m(x) = E[e^{v_i}|x_i] and
    \hat{m}'(x) is an estimated gradient of \hat{m}'(x) which should converge to m'(x)=dm(x)/d(x).
    \hat\beta is the vector of OLS estimators for \beta.
    """
    
    def __init__(self, endog, exog, interest=None, log_x=False, estimator_type='kernel', 
                 elasticity=False, kernel_params=None, nn_params=None, 
                 density_estimator='kernel', density_params=None, fe=None):
        """Initialize the doubly robust estimator."""

        # Initialize the base Model class
        super(DoublyRobustElasticityEstimator, self).__init__(endog, exog)

        # Store the original data
        self.original_endog = endog
        self.original_exog = exog
        self.fe = fe
        
        # Convert pandas objects to numpy arrays and store names
        if isinstance(endog, pd.Series):
            # Store the name but don't set endog_names (will be handled by Model.__init__)
            self._endog_name = endog.name
            endog = endog.values
        else:
            self._endog_name = 'y'
            
        if isinstance(exog, pd.DataFrame):
            self._exog_names = exog.columns.tolist()
            exog = exog.values
        else:
            self._exog_names = [f'x{i}' for i in range(exog.shape[1])]
        
        # Process fixed effects if provided
        if fe is not None:
            if isinstance(fe, list):
                if isinstance(fe[0], str):
                    self.fe_indices = self.original_exog.columns.get_indexer(['rail', 'distid'])
                else:
                    self.fe_indices = fe
            if isinstance(fe, int):
                self.fe_indices = [fe]
            if isinstance(fe, str):
                self.fe_indices = [self.original_exog.columns.get_loc(fe)]
            self._apply_fixed_effects()
        else:
            # If no fixed effects, just use the original data
            self.endog = endog
            self.exog = exog
        
        # Process interest variable(s)
        if interest is None:
            # If no specific interest variable, all variables are of interest
            self.interest = list(range(exog.shape[1]))
            self.all_variables = True
        else:
            self.all_variables = False
            if isinstance(interest, (list, tuple, np.ndarray)):
                # If a list of variables is provided
                self.interest = []
                for var in interest:
                    if isinstance(var, str) and var in self._exog_names:
                        self.interest.append(self._exog_names.index(var))
                    elif isinstance(var, (int, np.integer)):
                        self.interest.append(var)
                    else:
                        raise ValueError(f"Interest variable '{var}' not recognized")
            elif isinstance(interest, str) and interest in self._exog_names:
                self.interest = [self._exog_names.index(interest)]
            else:
                self.interest = [interest]
            
        # Process log_x
        if isinstance(log_x, (list, tuple, np.ndarray)):
            if len(log_x) != len(self.interest):
                raise ValueError("log_x must be a single bool or a list matching the length of interest")
            self.log_x = log_x
        else:
            # If a single value is provided, replicate it for all variables of interest
            self.log_x = [log_x] * len(self.interest)
            
        # Store other parameters
        self.estimator_type = estimator_type.lower()
        self.elasticity = elasticity
        self.density_estimator = density_estimator.lower()
        
        # Parameter dictionaries with defaults
        self.kernel_params = kernel_params or {}
        self.nn_params = nn_params or {}
        self.density_params = density_params or {}
        
        # Set default values for kernel parameters if not provided
        if self.estimator_type == 'kernel':
            self.kernel_params.setdefault('var_type', 'c' * exog.shape[1])
            self.kernel_params.setdefault('bw', 'cv_ls')
        
        # Set default values for density estimator parameters
        if self.density_estimator == 'kernel':
            self.density_params.setdefault('var_type', 'c' * exog.shape[1])
            self.density_params.setdefault('bw', 'normal_reference')
        elif self.density_estimator == 'nn':
            self.density_params.setdefault('hidden_layers', [64, 32])
            self.density_params.setdefault('activation', 'relu')
            self.density_params.setdefault('epochs', 100)
            self.density_params.setdefault('batch_size', 32)
        
        # Automatically detect variable types
        self._detect_variable_types()
        

    
    def _apply_fixed_effects(self):
        """Apply fixed effects transformation to the data."""
        # Extract the fixed effects columns

        if isinstance(self.original_exog, pd.DataFrame):
            fe_cols = self.original_exog.iloc[:, self.fe_indices].values
        else:
            fe_cols = self.original_exog[:, self.fe_indices]

        if fe_cols.ndim == 1:
            fe_cols = fe_cols.reshape(-1, 1)
        
        # Create dummy variables for fixed effects
        one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        dummy_vars = one_hot_encoder.fit_transform(fe_cols)
        
        # Create residual maker matrix
        residual_maker = np.eye(len(dummy_vars)) - dummy_vars @ np.linalg.pinv(
            np.transpose(dummy_vars) @ dummy_vars) @ np.transpose(dummy_vars)
        
        # Apply demeaning to both X and y
        if isinstance(self.original_endog, pd.Series) or isinstance(self.original_endog, pd.DataFrame):
            endog_to_use = np.log(self.original_endog.copy().values.reshape(-1, 1))
        else:
            endog_to_use = np.log(self.original_endog.copy().reshape(-1, 1))

        if isinstance(self.original_exog, pd.DataFrame):
            exog_to_use = self.original_exog.copy().values
        else:
            exog_to_use = self.original_exog.copy()

        combined = np.column_stack([exog_to_use, endog_to_use])
        demeaned = residual_maker @ combined
        
        # Update X and y with demeaned values
        self.exog = np.delete(demeaned[:, :-1], self.fe_indices, axis=1)
        self.endog = np.exp(demeaned[:, -1])
        
        # Update exog_names to remove fixed effect variables
        if hasattr(self, '_exog_names'):
            self._exog_names = [name for i, name in enumerate(self._exog_names) 
                              if i not in self.fe_indices]
        
        # Update interest index if needed
        if hasattr(self, 'interest') and isinstance(self.interest, int):
            # Adjust interest index after removing fixed effect columns
            adjusted_index = self.interest
            for fe_idx in sorted(self.fe_indices):
                if fe_idx < self.interest:
                    adjusted_index -= 1
            self.interest = adjusted_index
        
        print('Fixed effects applied. Variables have been demeaned.')
        
    def _detect_variable_types(self):
        """Detect binary and ordinal variables in the regressor matrix."""
        n_unique = np.array([len(np.unique(self.exog[:, i])) for i in range(self.exog.shape[1])])
        
        # Binary variables have 2 unique values
        self.binary_vars = np.where(n_unique == 2)[0]
        
        # Ordinal variables have between 3 and 10 unique values and are integers
        is_integer = np.array([np.all(np.equal(np.mod(self.exog[:, i], 1), 0)) 
                               for i in range(self.exog.shape[1])])
        self.ordinal_vars = np.where((n_unique > 2) & (n_unique <= 10) & is_integer)[0]
        
        # Update var_type in kernel_params based on detected variable types
        if self.estimator_type == 'kernel':
            var_type = ['c'] * self.exog.shape[1]
            for i in self.binary_vars:
                var_type[i] = 'u'
            for i in self.ordinal_vars:
                var_type[i] = 'o'
            self.kernel_params['var_type'] = ''.join(var_type)
        
        print(f"Detected {len(self.binary_vars)} binary variables: {self.binary_vars}")
        print(f"Detected {len(self.ordinal_vars)} ordinal variables: {self.ordinal_vars}")
    
    @property
    def exog_names(self):
        """Get the names of the explanatory variables."""
        return self._exog_names
    
    @property
    def endog_names(self):
        """Get the name of the response variable."""
        return self._endog_name
        
    def fit(self, weights=None, method='ols', bootstrap=False, bootstrap_reps=500, 
        bootstrap_method='pairs', **kwargs):
        """Fit the doubly robust estimator.
        
        Parameters
        ----------
        weights : array-like, optional
            Weights for weighted least squares, defaults to None
        method : str, optional
            Method to use for the parametric component, currently only 'ols' is supported
        bootstrap : bool, optional
            Whether to compute bootstrap standard errors, defaults to False
        bootstrap_reps : int, optional
            Number of bootstrap replications if bootstrap=True, defaults to 500
        bootstrap_method : str, optional
            Bootstrap method to use, either 'pairs' (resample observation pairs) or 
            'residuals' (resample residuals), defaults to 'pairs'
        **kwargs : dict
            Additional arguments to pass to the OLS fit method
            
        Returns
        -------
        DoublyRobustElasticityEstimatorResults
            Fitted model results object
        """

        kwargs.setdefault("cov_type", 'HC3')

        # Step 1: Fit the parametric model (OLS)
        if method == 'ols':
            if weights is None:
                ols_results = sm.OLS(np.log(self.endog), self.exog).fit(**kwargs)
            else:
                ols_results = sm.WLS(np.log(self.endog), self.exog, weights=weights).fit(**kwargs)
        else:
            raise ValueError(f"Method '{method}' not supported. Currently only 'ols' is available.")
            
        # Print OLS results for variables of interest
        if self.all_variables:
            print(f'OLS model fitted for all variables')
        elif len(self.interest) == 1:
            var_idx = self.interest[0]
            var_name = self.exog_names[var_idx]
            print(f'OLS model fitted, estimated beta for {var_name}: {ols_results.params[var_idx]:.4f}')
        else:
            print(f'OLS model fitted for {len(self.interest)} variables of interest')
            for var_idx in self.interest:
                var_name = self.exog_names[var_idx]
                print(f'  {var_name}: {ols_results.params[var_idx]:.4f}')
            
        # Step 2: Calculate residuals and exponentiate them
        residuals = ols_results.resid
        exp_residuals = np.exp(residuals)
        
        print('Estimating correction term...')
        
        # Step 3: Fit the nonparametric model for E[exp(u)|X]
        if self.estimator_type == 'kernel':
            nonparam_model = self._fit_kernel_model(exp_residuals)
        elif self.estimator_type == 'nn':
            nonparam_model = self._fit_nn_model(exp_residuals)
        elif self.estimator_type == 'ols':  # Add this option
            nonparam_model = self._fit_ols_model(exp_residuals)
        elif self.estimator_type == 'binary':
            # For binary estimator, check if any variables of interest are binary
            binary_interest = False
            for var_idx in self.interest:
                if var_idx in self.binary_vars:
                    binary_interest = True
                    break
                    
            if not binary_interest:
                warnings.warn("Binary estimator requested but no variables of interest are binary. "
                             "Consider using 'kernel', 'nn', or 'ols' estimator instead.")
            nonparam_model = self._fit_binary_model(exp_residuals)
        else:
            raise ValueError(f"Estimator type '{self.estimator_type}' not supported. "
                            f"Use 'kernel', 'nn', 'ols', or 'binary'.")
        
        # Step 4: Fit the density estimator if using neural network density estimation
        if self.density_estimator == 'nn':
            density_model = self._fit_nn_density_estimator(self.exog)
        else:
            density_model = None
            
        print('Correction term estimated')
        
        # Step 5: Create the results object
        results = DoublyRobustElasticityEstimatorResults(
            model=self,
            ols_results=ols_results,
            nonparam_model=nonparam_model,
            exp_residuals=exp_residuals,
            density_model=density_model
        )
        
        # Step 6: Compute bootstrap standard errors if requested
        if bootstrap:
            print(f'Computing bootstrap standard errors with {bootstrap_reps} replications...')
            results._bootstrap_standard_errors(bootstrap_reps, bootstrap_method, weights, kwargs)
            print('Bootstrap completed')
            
        return results
    
    def _fit_kernel_model(self, exp_residuals):
        """Fit a local linear kernel regression model for E[exp(u)|X]."""
        # Extract kernel parameters
        var_type = self.kernel_params.get('var_type', 'c' * self.exog.shape[1])
        bw = self.kernel_params.get('bw', 'cv_ls')
        
        # Normalize exponentiated residuals for numerical stability
        min_exp_residual = np.min(exp_residuals)
        normalized_exp_residuals = exp_residuals / min_exp_residual
        
        # Fit the kernel regression model
        kr_model = KernelReg(
            endog=normalized_exp_residuals,
            exog=self.exog,
            var_type=var_type,
            reg_type='ll',  # Local linear regression
            bw=bw
        )
        
        return KernelRegressionModel(kr_model, self.binary_vars, self.ordinal_vars, min_exp_residual)
    
    def _fit_nn_model(self, exp_residuals):
        """Fit a neural network model for E[exp(u)|X]."""
        # Import tensorflow only when needed
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required for neural network estimator. "
                             "Please install tensorflow package.")
        
        # Default neural network parameters
        defaults = {
            'num_layers': 3,
            'activation': 'relu',
            'num_units': 64,
            'optimizer': 'adam',
            'loss': 'mean_squared_error',
            'validation_split': 0.2,
            'batch_size': 64,
            'epochs': 100,
            'patience': 10,
            'verbose': 1
        }
        
        # Update defaults with user-provided parameters
        nn_params = {**defaults, **self.nn_params}
        
        # Create the neural network model
        model = self._create_nn_model(nn_params)
        
        # Normalize exponentiated residuals for numerical stability
        min_exp_residual = np.min(exp_residuals)
        normalized_exp_residuals = exp_residuals / min_exp_residual
        
        # Further normalize to prevent explosion in neural network training
        residuals_mean = np.mean(normalized_exp_residuals)
        residuals_std = np.std(normalized_exp_residuals)
        scaled_residuals = (normalized_exp_residuals - residuals_mean) / residuals_std
        
        # Prepare the inputs
        X_nn = self.exog.copy()
        
        # Convert to tensors
        X_tensor = tf.convert_to_tensor(X_nn, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(scaled_residuals, dtype=tf.float32)
        
        # Fit the model
        model.fit(
            x=X_tensor,
            y=y_tensor,
            validation_split=nn_params['validation_split'],
            batch_size=nn_params['batch_size'],
            epochs=nn_params['epochs'],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=nn_params['patience'],
                    restore_best_weights=True
                )
            ],
            verbose=nn_params['verbose']
        )
        
        # Create and return the NNRegressionModel
        return NNRegressionModel(model, residuals_mean, residuals_std, min_exp_residual, 
                                 self.binary_vars, self.ordinal_vars)
    
    def _fit_binary_model(self, exp_residuals):
        """Fit a binary model for E[exp(u)|X] when the regressor of interest is binary."""
        #TODO: NEEDS TO BE CORRECTED FOR FE ESTIMATION
        # This method needs to handle multiple binary variables of interest
        # Normalize exponentiated residuals for numerical stability
        min_exp_residual = np.min(exp_residuals)
        normalized_exp_residuals = exp_residuals / min_exp_residual
        
        # For binary model, we consider only binary variables in the interest list
        binary_interest = [var_idx for var_idx in self.interest if var_idx in self.binary_vars]
        
        # If no binary variables of interest, use the first variable in interest list
        if not binary_interest and self.interest:
            binary_interest = [self.interest[0]]
        
        # Fit binary models for all binary variables of interest
        binary_models = {}
        for var_idx in binary_interest:
            # Extract the binary variable
            x_binary = self.exog[:, var_idx]
            
            # Calculate the mean of normalized exp_residuals for each value of the binary variable
            eu_0 = np.mean(normalized_exp_residuals[x_binary == 0])
            eu_1 = np.mean(normalized_exp_residuals[x_binary == 1])
            
            # Store the model
            binary_models[var_idx] = (eu_0, eu_1)
        
        # Create a binary regression model that handles multiple variables
        return MultiBinaryRegressionModel(binary_models, min_exp_residual, 
                                         self.binary_vars, self.ordinal_vars)

    def _fit_ols_model(self, exp_residuals):
        """Fit a polynomial OLS model for E[exp(u)|X].
        
        Parameters
        ----------
        exp_residuals : array-like
            Exponentiated residuals from the parametric model
            
        Returns
        -------
        OLSRegressionModel
            Fitted OLS regression model
        """
        # Get polynomial degree from parameters (default is 3)
        degree = self.kernel_params.get('degree', 3)
        
        # Normalize exponentiated residuals for numerical stability
        min_exp_residual = np.min(exp_residuals)
        normalized_exp_residuals = exp_residuals / min_exp_residual
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(self.exog)
        
        # Fit OLS model on polynomial features
        ols_model = sm.OLS(normalized_exp_residuals, X_poly).fit()
        
        print(f'OLS correction model fitted with polynomial degree {degree}')
        print(f'R-squared: {ols_model.rsquared:.4f}')
        
        return OLSRegressionModel(ols_model, poly, min_exp_residual, 
                                 self.binary_vars, self.ordinal_vars)
    
    def _create_nn_model(self, nn_params):
        """Create a neural network model based on the provided parameters."""
        import tensorflow as tf
        
        num_layers = nn_params['num_layers']
        activation = nn_params['activation']
        num_units = nn_params['num_units']
        optimizer = nn_params['optimizer']
        loss = nn_params['loss']
        
        # Handle num_units as either int or list
        if isinstance(num_units, int):
            num_units = [num_units] * num_layers
        elif len(num_units) != num_layers:
            raise ValueError("num_units must be an int or a list of length num_layers")
        
        # Create the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.exog.shape[1],)))
        for i in range(num_layers):
            model.add(tf.keras.layers.Dense(num_units[i], activation=activation))
        model.add(tf.keras.layers.Dense(1))
        
        # Compile the model
        model.compile(optimizer=optimizer, loss=loss)
        
        return model


import numpy as np
import pandas as pd
from statsmodels.base.model import Results
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import joblib


class DoublyRobustElasticityEstimatorResults(Results):
    """
    Results from the DoublyRobustElasticityEstimator with corrected asymptotic variance.
    
    This class stores and presents results from the doubly robust elasticity estimator,
    providing methods to access various statistics, visualizations, and tests with
    properly implemented multivariate asymptotic variance calculations.
    
    Parameters
    ----------
    model : DoublyRobustElasticityEstimator
        The estimator model
    ols_results : RegressionResults
        Results from the initial OLS model
    nonparam_model : RegressionModel
        The fitted nonparametric model
    exp_residuals : array_like
        Exponentiated residuals from the parametric model
    density_model : NNDensityModel or None, optional
        The fitted density model for high-dimensional X
    bootstrap : bool, optional
        Whether bootstrap standard errors have been computed
    bootstrap_reps : int, optional
        Number of bootstrap replications
    bootstrap_estimates : array-like, optional
        Bootstrap estimates
    kernel_type : str, default 'gaussian'
        Type of kernel used for nonparametric estimation
    density_bandwidth_method : str, default 'scott'
        Method for selecting density estimation bandwidth
        
    Attributes
    ----------
    model : DoublyRobustElasticityEstimator
        The fitted estimator model
    ols_results : RegressionResults
        OLS regression results
    nonparam_model : RegressionModel
        Fitted nonparametric model for correction term
    exp_residuals : ndarray
        Exponentiated residuals from parametric model
    beta_hat : list or ndarray
        OLS estimates for variables of interest
    m_hat : ndarray
        Predicted values from nonparametric model
    m_prime_hat : dict
        Derivatives from nonparametric model for each variable
    correction : dict
        Correction terms for each variable
    estimator_values : dict
        Final estimator values for each variable
    kernel_constants : dict
        Computed kernel constants for asymptotic variance
    """
    
    def __init__(self, model, ols_results, nonparam_model, exp_residuals, 
                 density_model=None, bootstrap=False, bootstrap_reps=None, 
                 bootstrap_estimates=None, kernel_type='gaussian', 
                 density_bandwidth_method='scott'):
        """Initialize the results object with corrected asymptotic variance capabilities."""
        self.model = model
        self.ols_results = ols_results
        self.nonparam_model = nonparam_model
        self.exp_residuals = exp_residuals
        self.interest = model.interest
        self.all_variables = model.all_variables
        self.density_model = density_model
        
        # Asymptotic variance parameters
        self.kernel_type = kernel_type.lower()
        self.density_bandwidth_method = density_bandwidth_method
        
        # Extract key dimensions
        self.n_obs = model.exog.shape[0]
        self.n_vars = model.exog.shape[1]
        
        # Bootstrap-related attributes
        self.bootstrap = bootstrap
        self.bootstrap_reps = bootstrap_reps
        self.bootstrap_estimates = bootstrap_estimates
        
        # Extract the OLS estimate of the parameter(s) of interest
        if self.all_variables:
            self.beta_hat = ols_results.params
        else:
            self.beta_hat = [ols_results.params[idx] for idx in self.interest]
        
        # Initialize asymptotic variance components
        self._initialize_asymptotic_variance()
        
        # Compute correction terms for all data points
        self._compute_corrections()
        
        # Initialize the base Results class
        super(DoublyRobustElasticityEstimatorResults, self).__init__(model, ols_results.params)
    
    def _initialize_asymptotic_variance(self):
        """
        Initialize components needed for asymptotic variance calculation.
        
        This method computes kernel constants and sets up the conditional variance
        model that will be used throughout the asymptotic variance calculations.
        """
        # Compute kernel constants
        self.kernel_constants = self._compute_kernel_constants()
        
        # Initialize conditional variance model
        self._conditional_variance_model = None
        self._fit_conditional_variance_model()
    
    def _compute_kernel_constants(self):
        """
        Compute kernel constants for multivariate asymptotic variance calculation.
        
        For product kernels K(u) = ∏_{j=1}^d K_0(u_j), computes:
        - ν₀(K) = (ν₀(K₀))^d  
        - 𝒦₀ = ν₀(K) I_d
        - 𝒦₁ = [ν₂(K₀)(ν₀(K₀))^{d-1} / μ₂(K₀)²] I_d
        
        Returns
        -------
        dict
            Dictionary containing all kernel constants needed for variance calculations
        """
        d = self.n_vars
        
        if self.kernel_type == 'gaussian':
            # Theoretical values for standard Gaussian kernel K₀(u) = (2π)^{-1/2} exp(-u²/2)
            nu_0_K0 = 1 / (2 * np.sqrt(np.pi))  # ≈ 0.2821
            mu_2_K0 = 1.0
            nu_2_K0 = 3 / (4 * np.sqrt(np.pi))  # ≈ 0.4239
            
        elif self.kernel_type == 'epanechnikov':
            # Theoretical values for Epanechnikov kernel K₀(u) = (3/4)(1-u²)𝟙{|u|≤1}
            nu_0_K0 = 3/5
            mu_2_K0 = 1/5  
            nu_2_K0 = 3/35
            
        else:
            warnings.warn(f"Unknown kernel type '{self.kernel_type}'. Using Gaussian defaults.")
            nu_0_K0 = 1 / (2 * np.sqrt(np.pi))
            mu_2_K0 = 1.0
            nu_2_K0 = 3 / (4 * np.sqrt(np.pi))
        
        # Compute multivariate kernel constants
        K_0_scalar = nu_0_K0 ** d
        K_0_matrix = K_0_scalar * np.eye(d)
        
        K_1_scalar = (nu_2_K0 * (nu_0_K0 ** (d-1))) / (mu_2_K0 ** 2)
        K_1_matrix = K_1_scalar * np.eye(d)
        
        return {
            'nu_0_K0': nu_0_K0,
            'nu_2_K0': nu_2_K0, 
            'mu_2_K0': mu_2_K0,
            'K_0_scalar': K_0_scalar,
            'K_0_matrix': K_0_matrix,
            'K_1_scalar': K_1_scalar,
            'K_1_matrix': K_1_matrix
        }
    
    def _fit_conditional_variance_model(self):
        """
        Fit nonparametric model for conditional variance σ²_w(x) = Var(e^u|X=x).
        
        Uses the same nonparametric method as the main model to estimate
        E[w²|X] where w = e^u, then computes σ²_w(x) = E[w²|X=x] - (E[w|X=x])².
        """
        if self.model.estimator_type == 'kernel':
        # Use same kernel parameters as main model
            var_type = self.model.kernel_params.get('var_type', 'c' * self.n_vars)
            bw = self.model.kernel_params.get('bw', 'cv_ls')
            
            exp_residuals_squared = self.exp_residuals ** 2
            
            from statsmodels.nonparametric.kernel_regression import KernelReg
            self._conditional_variance_model = KernelReg(
                endog=exp_residuals_squared,
                exog=self.model.exog,
                var_type=var_type,
                reg_type='ll',
                bw=bw
            )
            
        elif self.model.estimator_type == 'ols':
            # Use polynomial features
            degree = self.model.kernel_params.get('degree', 3)
            self._poly_variance = PolynomialFeatures(degree=degree, include_bias=False)
            
            exp_residuals_squared = self.exp_residuals ** 2
            X_poly = self._poly_variance.fit_transform(self.model.exog)
            self._conditional_variance_model = sm.OLS(exp_residuals_squared, X_poly).fit()
            
        elif self.model.estimator_type == 'nn':
            # NEW: Proper neural network conditional variance estimation
            print("Fitting neural network for conditional variance estimation...")
            self._conditional_variance_model = self._fit_nn_conditional_variance_model()
            print("Neural network conditional variance model fitted successfully!")
            
        else:
            # For binary and other methods, use simple kernel
            exp_residuals_squared = self.exp_residuals ** 2
            var_type = 'c' * self.n_vars
            
            from statsmodels.nonparametric.kernel_regression import KernelReg
            self._conditional_variance_model = KernelReg(
                endog=exp_residuals_squared,
                exog=self.model.exog,
                var_type=var_type,
                reg_type='ll',
                bw='cv_ls'
            )



    def _fit_nn_conditional_variance_model(self):
        """
        Fit neural network model for conditional variance estimation.
        
        This method fits a dedicated neural network to estimate E[w²|X] where w = e^u,
        using similar architecture to the main neural network but optimized for variance.
        
        Returns
        -------
        NNConditionalVarianceModel
            Fitted neural network model for conditional variance estimation
            
        Notes
        -----
        The architecture is slightly simplified compared to the main model since
        variance estimation is often less complex than mean estimation. Includes
        regularization and early stopping to prevent overfitting.
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required for neural network estimator. "
                             "Please install tensorflow package.")
        
        # Get squared exponentiated residuals
        exp_residuals_squared = self.exp_residuals ** 2
        
        # Use same neural network parameters as main model, with some adjustments
        nn_params = dict(self.model.nn_params)  # Create a copy
        
        # Default parameters specifically for conditional variance estimation
        nn_params.setdefault('num_layers', 3)
        nn_params.setdefault('activation', 'relu')  
        nn_params.setdefault('num_units', 64)
        nn_params.setdefault('optimizer', 'adam')
        nn_params.setdefault('loss', 'mean_squared_error')
        nn_params.setdefault('validation_split', 0.2)
        nn_params.setdefault('batch_size', 64)
        nn_params.setdefault('epochs', 100)
        nn_params.setdefault('patience', 10)
        nn_params.setdefault('verbose', 0)  # Less verbose for conditional variance
        
        # Reduce complexity slightly for variance estimation (often less complex than mean)
        if isinstance(nn_params['num_units'], list):
            # Reduce each layer size by ~25%
            nn_params['num_units'] = [max(16, int(0.75 * units)) for units in nn_params['num_units']]
        else:
            nn_params['num_units'] = max(16, int(0.75 * nn_params['num_units']))
        
        # Create the neural network model for conditional variance
        variance_model = self._create_nn_variance_model(nn_params)
        
        # Normalize squared exponentiated residuals for numerical stability
        min_exp_residual_sq = np.min(exp_residuals_squared)
        if min_exp_residual_sq <= 0:
            min_exp_residual_sq = 1e-10  # Prevent division by zero
            
        normalized_exp_residuals_sq = exp_residuals_squared / min_exp_residual_sq
        
        # Further standardize to prevent explosion in neural network training
        residuals_mean = np.mean(normalized_exp_residuals_sq)
        residuals_std = np.std(normalized_exp_residuals_sq)
        
        # Prevent division by zero
        if residuals_std < 1e-8:
            residuals_std = 1.0
            warnings.warn("Very low variance in squared residuals. Check for homoskedasticity.")
        
        scaled_residuals_sq = (normalized_exp_residuals_sq - residuals_mean) / residuals_std
        
        # Prepare the inputs
        X_nn = self.model.exog.copy()
        
        # Convert to tensors
        X_tensor = tf.convert_to_tensor(X_nn, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(scaled_residuals_sq, dtype=tf.float32)
        
        # Set up callbacks for robust training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=nn_params['patience'],
                restore_best_weights=True,
                verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(5, nn_params['patience'] // 2),
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        # Fit the model with error handling
        try:
            history = variance_model.fit(
                x=X_tensor,
                y=y_tensor,
                validation_split=nn_params['validation_split'],
                batch_size=nn_params['batch_size'],
                epochs=nn_params['epochs'],
                callbacks=callbacks,
                verbose=nn_params['verbose']
            )
            
            # Check if training was successful
            final_loss = history.history['loss'][-1]
            if np.isnan(final_loss) or np.isinf(final_loss):
                raise ValueError("Neural network training failed - loss is NaN or infinite")
                
            # Check for reasonable convergence
            if len(history.history['loss']) < 5:
                warnings.warn("Neural network training stopped very early. Results may be unreliable.")
                
        except Exception as e:
            warnings.warn(f"Neural network conditional variance training failed: {e}. "
                         f"Falling back to kernel method.")
            # Fall back to kernel method
            return self._fit_kernel_conditional_variance_fallback()
        
        # Create and return the NNConditionalVarianceModel
        # Account for the scaling factor from the original model
        scaling_factor = min_exp_residual_sq
        
        return NNConditionalVarianceModel(
            model=variance_model,
            mean=residuals_mean,
            std=residuals_std, 
            scaling_factor=scaling_factor,
            binary_vars=self.model.binary_vars,
            ordinal_vars=self.model.ordinal_vars
        )


    def _create_nn_variance_model(self, nn_params):
        """
        Create a neural network model optimized for conditional variance estimation.
        
        Parameters
        ----------
        nn_params : dict
            Neural network parameters
            
        Returns
        -------
        tensorflow.keras.Model
            Compiled neural network model for variance estimation
            
        Notes
        -----
        The architecture includes regularization and dropout to prevent overfitting,
        which is particularly important for variance estimation where the signal
        may be weaker than for mean estimation.
        """
        import tensorflow as tf
        
        num_layers = nn_params['num_layers']
        activation = nn_params['activation']
        num_units = nn_params['num_units']
        optimizer = nn_params['optimizer']
        loss = nn_params['loss']
        
        # Handle num_units as either int or list
        if isinstance(num_units, int):
            num_units = [num_units] * num_layers
        elif len(num_units) != num_layers:
            raise ValueError("num_units must be an int or a list of length num_layers")
        
        # Create the model with appropriate architecture for variance estimation
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.model.exog.shape[1],)))
        
        # Add hidden layers with regularization
        for i in range(num_layers):
            model.add(tf.keras.layers.Dense(
                num_units[i], 
                activation=activation,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                kernel_initializer='he_normal'  # Good for ReLU activations
            ))
            
            # Add batch normalization for training stability
            model.add(tf.keras.layers.BatchNormalization())
            
            # Add dropout for regularization (except last hidden layer)
            if i < num_layers - 1:
                model.add(tf.keras.layers.Dropout(0.1))
        
        # Output layer - no activation since variance can be any positive value
        # We'll enforce non-negativity in the conditional variance calculation
        model.add(tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform'))
        
        # Compile the model with appropriate settings for variance estimation
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(
                learning_rate=0.001, 
                clipnorm=1.0,  # Gradient clipping for stability
                beta_1=0.9,
                beta_2=0.999
            )
        else:
            opt = optimizer
            
        model.compile(
            optimizer=opt, 
            loss=loss, 
            metrics=['mae', 'mse']
        )
        
        return model

    def _fit_kernel_conditional_variance_fallback(self):
        """
        Fallback method using kernel regression when neural network fails.
        
        Returns
        -------
        KernelReg
            Fitted kernel regression model for conditional variance
            
        Notes
        -----
        This fallback ensures robustness when neural network training fails
        due to numerical issues or other problems.
        """
        from statsmodels.nonparametric.kernel_regression import KernelReg
        
        exp_residuals_squared = self.exp_residuals ** 2
        var_type = 'c' * self.n_vars
        
        print("Using kernel regression fallback for conditional variance...")
        
        return KernelReg(
            endog=exp_residuals_squared,
            exog=self.model.exog,
            var_type=var_type,
            reg_type='ll',
            bw='cv_ls'
        )

    
    def _estimate_conditional_variance(self, x):
        """
        Estimate conditional variance σ²_w(x) = Var(e^u|X=x) at point x.
        
        Parameters
        ----------
        x : array_like of shape (1, d) or (d,)
            Point at which to evaluate conditional variance
            
        Returns
        -------
        float
            Estimated conditional variance σ²_w(x)
        """
        x = np.atleast_2d(x)
        if x.shape[0] != 1:
            raise ValueError("x must be a single point of shape (1, d)")
        
        # Get E[w²|X=x]
        if self.model.estimator_type == 'ols' and hasattr(self, '_poly_variance'):
            x_poly = self._poly_variance.transform(x)
            m_w2_x = self._conditional_variance_model.predict(x_poly)[0]
        else:
            m_w2_x = self._conditional_variance_model.fit(x)[0][0]
        
        # Get E[w|X=x] = m(x)
        m_x = self.nonparam_model.predict(x)[0]
        
        # Compute conditional variance with non-negativity constraint
        conditional_var = max(0.0, m_w2_x - m_x**2)
        
        return conditional_var



    def _estimate_conditional_variance_nn(self, x):
        """
        Estimate conditional variance using neural network models.
        
        Parameters
        ----------
        x : array_like of shape (1, d)
            Point at which to evaluate conditional variance
            
        Returns
        -------
        float
            Estimated conditional variance σ²_w(x)
            
        Notes
        -----
        Uses the dedicated neural network model for E[w²|X] and combines it
        with the main model's E[w|X] to compute the conditional variance.
        """
        # Check if we have a neural network conditional variance model
        if hasattr(self._conditional_variance_model, 'fit'):
            # Neural network case
            try:
                # Get E[w²|X=x] from the NN variance model
                m_w2_x = self._conditional_variance_model.fit(x)[0]
                
                # Get E[w|X=x] = m(x) from the main model
                m_x = self.nonparam_model.predict(x)[0]
                
                # Compute conditional variance: Var(w|X=x) = E[w²|X=x] - (E[w|X=x])²
                conditional_var = m_w2_x - m_x**2
                
                # Ensure non-negativity (numerical precision issues can cause small negative values)
                conditional_var = max(0.0, conditional_var)
                
                # Additional sanity check
                if conditional_var > 1000 * np.var(self.exp_residuals):
                    warnings.warn("Conditional variance estimate seems unusually large. "
                                 "Check neural network training.")
                    # Fall back to a reasonable value
                    conditional_var = min(conditional_var, 10 * np.var(self.exp_residuals))
                
                return conditional_var
                
            except Exception as e:
                warnings.warn(f"Neural network conditional variance prediction failed: {e}. "
                             f"Using fallback method.")
                # Fall back to kernel method if NN prediction fails
                return self._estimate_conditional_variance_kernel_fallback(x)
        else:
            # Fallback if model doesn't have NN interface
            return self._estimate_conditional_variance_kernel_fallback(x)


    def _estimate_conditional_variance_kernel_fallback(self, x):
        """
        Fallback conditional variance estimation using kernel methods.
        
        Parameters
        ----------
        x : array_like of shape (1, d)
            Point at which to evaluate conditional variance
            
        Returns
        -------
        float
            Estimated conditional variance using kernel fallback
            
        Notes
        -----
        This method provides a robust fallback when neural network prediction
        fails or produces unreasonable results.
        """
        # Use kernel regression as fallback
        exp_residuals_squared = self.exp_residuals ** 2
        
        from statsmodels.nonparametric.kernel_regression import KernelReg
        var_type = 'c' * self.n_vars
        
        kernel_var_model = KernelReg(
            endog=exp_residuals_squared,  
            exog=self.model.exog,
            var_type=var_type,
            reg_type='ll',
            bw='cv_ls'
        )
        
        m_w2_x = kernel_var_model.fit(x)[0][0]
        m_x = self.nonparam_model.predict(x)[0]
        
        return max(0.0, m_w2_x - m_x**2)


    def _estimate_conditional_variance_original(self, x):
        """
        Original conditional variance estimation method for non-NN cases.
        
        Parameters
        ----------
        x : array_like of shape (1, d)
            Point at which to evaluate conditional variance
            
        Returns
        -------
        float
            Estimated conditional variance using the original method
            
        Notes
        -----
        This method handles kernel, OLS, and other non-neural network estimators
        using the appropriate fitted conditional variance models.
        """
        # Get E[w²|X=x] using the appropriate method
        if self.model.estimator_type == 'ols' and hasattr(self, '_poly_variance'):
            x_poly = self._poly_variance.transform(x)
            m_w2_x = self._conditional_variance_model.predict(x_poly)[0]
        else:
            # Kernel or other methods
            m_w2_x = self._conditional_variance_model.fit(x)[0][0]
        
        # Get E[w|X=x] = m(x)
        m_x = self.nonparam_model.predict(x)[0]
        
        # Compute conditional variance with non-negativity constraint
        conditional_var = max(0.0, m_w2_x - m_x**2)
        
        return conditional_var

    
    def _estimate_density(self, x):
        """
        Estimate density f_X(x) at point x using kernel density estimation.
        
        Parameters
        ----------
        x : array_like of shape (1, d) or (d,)
            Point at which to evaluate density
            
        Returns
        -------
        float
            Estimated density f_X(x)
        """
        x = np.atleast_2d(x)
        
        # Set up KDE parameters
        var_type = 'c' * self.n_vars
        
        # Create KDE object
        kde = KDEMultivariate(
            data=self.model.exog,
            var_type=var_type,
            bw=self.density_bandwidth_method
        )
        
        return kde.pdf(x.flatten())
    
    def _get_bandwidth(self):
        """
        Get bandwidth used in the nonparametric model.
        
        Returns
        -------
        float
            Bandwidth parameter. For multivariate case, returns mean bandwidth.
        """
        if hasattr(self.nonparam_model, 'bandwidth') and self.nonparam_model.bandwidth is not None:
            bandwidth = self.nonparam_model.bandwidth
            # Convert to scalar if needed (take mean for multivariate)
            if hasattr(bandwidth, '__len__') and len(bandwidth) > 1:
                return np.mean(bandwidth)
            else:
                return float(bandwidth)
        else:
            # Rule of thumb bandwidth for d-dimensional case
            return self.n_obs ** (-1 / (4 + self.n_vars))
    
    def _compute_corrections(self):
        """
        Compute correction terms for all data points in the sample.
        
        This method calculates the nonparametric corrections and final estimator
        values for each variable of interest.
        """
        # Get the values of m(x) for all observations
        self.m_hat = self.nonparam_model.predict(self.model.exog)
        
        # Initialize dictionaries to store results for each variable of interest
        self.m_prime_hat = {}
        self.correction = {}
        self.estimator_values = {}
        
        # Loop through variables of interest
        for i, var_idx in enumerate(self.interest):
            # Get derivative for this variable
            self.m_prime_hat[var_idx] = self.nonparam_model.derivative(self.model.exog, var_idx)
            
            if var_idx in self.model.binary_vars:
                # For binary variables, use the formula: e^{beta} * (m1/m0) - 1
                # Get beta coefficient
                if self.all_variables:
                    beta_hat = self.beta_hat[var_idx]
                else:
                    beta_hat = self.beta_hat[i]
                
                # Get m0 and m1 values
                X_0 = self.model.exog.copy()
                X_1 = self.model.exog.copy()
                X_0[:, var_idx] = 0
                X_1[:, var_idx] = 1
                m_0 = self.nonparam_model.predict(X_0)
                m_1 = self.nonparam_model.predict(X_1)
                
                # For binary variables, the semi-elasticity is e^{beta} * (m1/m0) - 1
                # This is constant for all observations
                semi_elasticity = np.exp(beta_hat) * (m_1 / m_0) - 1
                self.estimator_values[var_idx] = semi_elasticity
                
                # Store correction term for consistency (though not used for binary)
                self.correction[var_idx] = self.m_prime_hat[var_idx] / self.m_hat
            else:
                # For continuous variables, use the original logic
                # Compute the correction term
                self.correction[var_idx] = self.m_prime_hat[var_idx] / self.m_hat
                
                # Compute the doubly robust estimator values
                if self.model.elasticity and self.model.log_x[i]:
                    # For elasticity with log(x)
                    self.estimator_values[var_idx] = self.beta_hat[i] + self.correction[var_idx]
                elif not self.model.elasticity and not self.model.log_x[i]:
                    # For semi-elasticity with untransformed x
                    self.estimator_values[var_idx] = self.beta_hat[i] + self.correction[var_idx]
                elif self.model.elasticity and not self.model.log_x[i]:
                    # For elasticity with untransformed x
                    self.estimator_values[var_idx] = self.beta_hat[i] + self.correction[var_idx] * self.model.exog[:, var_idx]
                else:  # not self.model.elasticity and self.model.log_x[i]
                    # For semi-elasticity with log(x)
                    self.estimator_values[var_idx] = self.beta_hat[i] + self.correction[var_idx] / self.model.exog[:, var_idx]

    def asymptotic_variance_matrix(self, x):
        """
        Compute full d×d asymptotic variance-covariance matrix at point x.
        
        Parameters
        ----------
        x : array_like of shape (1, d) or (d,)
            Point at which to compute asymptotic variance matrix
            
        Returns
        -------
        ndarray of shape (d, d)
            Asymptotic variance-covariance matrix AVar(ẑ(x))
            
        Notes
        -----
        Implements the corrected multivariate formula:
        
        .. math::
            \\widehat{AVar}(\\hat{z}(x)) = \\frac{\\hat{\\sigma}_w^2(x)}{N \\hat{f}_X(x) h^{d+2} \\hat{m}(x)^4} 
            \\left[ \\hat{m}(x)^2 \\mathcal{K}_1 + h^2 \\nabla \\hat{m}(x)(\\nabla \\hat{m}(x))^T \\mathcal{K}_0 \\right]
        """
        x = np.atleast_2d(x)
        if x.shape[0] != 1:
            raise ValueError("x must be a single point of shape (1, d)")
        
        # Get required components
        sigma_w2_x = self._estimate_conditional_variance(x)
        f_x = self._estimate_density(x)
        h = self._get_bandwidth()
        m_x = self.nonparam_model.predict(x)[0]
        
        # Get gradient vector ∇m(x)
        grad_m_x = np.zeros(self.n_vars)
        for j in range(self.n_vars):
            grad_m_x[j] = self.nonparam_model.derivative(x, j)[0]
        
        # Extract kernel constants
        K_0_matrix = self.kernel_constants['K_0_matrix']
        K_1_matrix = self.kernel_constants['K_1_matrix']
        
        # Compute the two terms
        # Term 1: m̂(x)² 𝒦₁ / (nh^{d+2})
        term1 = (m_x**2 / (self.n_obs * h**(self.n_vars + 2))) * K_1_matrix
        
        # Term 2: h² ∇m̂(x)(∇m̂(x))ᵀ 𝒦₀ / (nh^d)  
        grad_outer = np.outer(grad_m_x, grad_m_x)
        term2 = (h**2 / (self.n_obs * h**self.n_vars)) * grad_outer @ K_0_matrix
        
        # Combine terms with common factor
        common_factor = sigma_w2_x / (f_x * m_x**4)
        
        avar_matrix = common_factor * (term1 + term2)
        
        return avar_matrix
    
    def asymptotic_variance(self, x, variable_idx=None):
        """
        Compute scalar asymptotic variance for specific variable at point x.
        
        Parameters
        ----------
        x : array_like of shape (1, d) or (d,)
            Point at which to compute asymptotic variance
        variable_idx : int or str, optional
            Index or name of variable for which to compute variance.
            If None, uses first variable of interest.
            
        Returns
        -------
        float
            Asymptotic variance AVar(ẑⱼ(x)) for variable j
        """
        # Handle variable selection
        if variable_idx is None:
            variable_idx = self.interest[0] if isinstance(self.interest, list) else self.interest
        elif isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        
        if not (0 <= variable_idx < self.n_vars):
            raise ValueError(f"variable_idx must be between 0 and {self.n_vars-1}")
        
        x = np.atleast_2d(x)
        if x.shape[0] != 1:
            raise ValueError("x must be a single point of shape (1, d)")
        
        # Get required components  
        sigma_w2_x = self._estimate_conditional_variance(x)
        f_x = self._estimate_density(x)
        h = self._get_bandwidth()
        m_x = self.nonparam_model.predict(x)[0]
        m_prime_x = self.nonparam_model.derivative(x, variable_idx)[0]
        
        # Extract scalar kernel constants
        K_0_scalar = self.kernel_constants['K_0_scalar']
        K_1_scalar = self.kernel_constants['K_1_scalar']
        
        # Compute scalar version of the formula
        # Term 1: m̂(x)² K₁ / (nh^{d+2})
        term1 = (m_x**2 * K_1_scalar) / (self.n_obs * h**(self.n_vars + 2))
        
        # Term 2: h² (∂m̂/∂xⱼ)² K₀ / (nh^d)
        term2 = (h**2 * m_prime_x**2 * K_0_scalar) / (self.n_obs * h**self.n_vars)
        
        # Common factor
        common_factor = sigma_w2_x / (f_x * m_x**4)
        
        avar_scalar = common_factor * (term1 + term2)
        
        return avar_scalar
    
    def asymptotic_standard_error_at_point(self, x, variable_idx=None):
        """
        Compute asymptotic standard error at point x.
        
        Parameters
        ----------
        x : array_like of shape (1, d) or (d,)
            Point at which to compute standard error
        variable_idx : int or str, optional
            Variable for which to compute standard error.
            If None, uses first variable of interest.
            
        Returns
        -------
        float
            Asymptotic standard error
        """
        variance = self.asymptotic_variance(x, variable_idx)
        return np.sqrt(variance)
    
    def asymptotic_standard_error_at_average(self, variable_idx=None):
        """
        Compute asymptotic standard error at average values of regressors.
        
        Parameters
        ----------
        variable_idx : int or str, optional
            Variable for which to compute standard error.
            If None, uses first variable of interest.
            
        Returns
        -------
        float
            Asymptotic standard error at average X
        """
        X_mean = np.mean(self.model.exog, axis=0).reshape(1, -1)
        return self.asymptotic_standard_error_at_point(X_mean, variable_idx)
    
    def asymptotic_confidence_interval(self, x, variable_idx=None, alpha=0.05):
        """
        Compute asymptotic confidence interval at point x.
        
        Parameters  
        ----------
        x : array_like of shape (1, d) or (d,)
            Point at which to compute confidence interval
        variable_idx : int or str, optional
            Variable for confidence interval. If None, uses first variable of interest.
        alpha : float, default 0.05
            Significance level (1-α confidence level)
            
        Returns
        -------
        tuple of float
            (lower_bound, upper_bound) for confidence interval
        """
        # Handle variable selection
        if variable_idx is None:
            variable_idx = self.interest[0] if isinstance(self.interest, list) else self.interest
        elif isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        
        estimate = self.estimate_at_point(x, variable_idx)
        std_error = self.asymptotic_standard_error_at_point(x, variable_idx)
        z_value = stats.norm.ppf(1 - alpha/2)
        
        lower_bound = estimate - z_value * std_error
        upper_bound = estimate + z_value * std_error
        
        return (lower_bound, upper_bound)

    # Keep all the existing methods but update the ones that use asymptotic variance
    def std_error_at_average(self, variable_idx=None):
        """
        Calculate the standard error of the estimator at average values.
        
        Parameters
        ----------
        variable_idx : int or str, optional
            Variable for which to compute standard error.
            
        Returns
        -------
        float
            Standard error at average X values
        """
        return self.asymptotic_standard_error_at_average(variable_idx)
    
    def conf_int(self, alpha=0.05, point=None, variable_idx=None):
        """
        Calculate confidence intervals for the estimator at a specific point.
        
        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for (1-α)×100% confidence interval
        point : array_like of shape (1, d), optional
            Point at which to compute confidence interval. 
            If None, uses average values of regressors.
        variable_idx : int or str, optional
            Variable for confidence interval. If None, uses first variable of interest.
            
        Returns
        -------
        tuple of float
            (lower_bound, upper_bound) for the confidence interval
        """
        if point is None:
            point = np.mean(self.model.exog, axis=0).reshape(1, -1)
            
        return self.asymptotic_confidence_interval(point, variable_idx, alpha)

    # Keep all other existing methods unchanged (bootstrap, plotting, etc.)
    def _bootstrap_standard_errors(self, bootstrap_reps, bootstrap_method, weights, kwargs):
        """
        Compute bootstrap standard errors.
        
        Parameters
        ----------
        bootstrap_reps : int
            Number of bootstrap replications
        bootstrap_method : str
            Bootstrap method to use: 'pairs' only (residuals method removed)
        weights : array-like or None
            Weights for WLS if provided
        kwargs : dict
            Additional arguments for the OLS fit method
        """
        if bootstrap_method != 'pairs':
            raise ValueError(f"Bootstrap method '{bootstrap_method}' not supported. "
                            f"Only 'pairs' bootstrap is available.")
        
        n = len(self.model.endog)
        
        # Initialize storage for bootstrap estimates
        if self.all_variables:
            var_indices = self.interest
        else:
            var_indices = self.interest
            
        bootstrap_estimates_dict = {var_idx: np.zeros((bootstrap_reps, 4)) 
                                   for var_idx in var_indices}
        
        for i in range(bootstrap_reps):
            if i > 0 and i % 100 == 0:
                print(f"  Completed {i} bootstrap replications")
                
            # Resample pairs (x_i, y_i)
            indices = np.random.choice(n, size=n, replace=True)
            bs_exog = self.model.exog[indices]
            bs_endog = self.model.endog[indices]
            
            # Create a new estimator with the resampled data
            bs_model = DoublyRobustElasticityEstimator(
                endog=bs_endog,
                exog=bs_exog,
                interest=self.model.interest,
                log_x=self.model.log_x,
                estimator_type=self.model.estimator_type,
                elasticity=self.model.elasticity,
                kernel_params=self.model.kernel_params,
                nn_params=self.model.nn_params,
                density_estimator=self.model.density_estimator,
                density_params=self.model.density_params
            )
            
            # Fit the bootstrapped model
            bs_results = bs_model.fit(weights=weights, bootstrap=False, **kwargs)
            
            # Store estimates from this bootstrap replication for each variable
            X_mean = np.mean(self.model.exog, axis=0).reshape(1, -1)
            for var_idx in var_indices:
                # Find the position of this variable in the bootstrapped results
                if bs_results.all_variables:
                    var_pos = list(bs_results.interest).index(var_idx)
                    beta_hat = bs_results.beta_hat[var_idx]
                else:
                    var_pos = bs_results.interest.index(var_idx)
                    beta_hat = bs_results.beta_hat[var_pos]
                
                original_X_mean = np.mean(self.model.exog, axis=0).reshape(1, -1)

                bootstrap_estimates_dict[var_idx][i, 0] = beta_hat  # OLS estimate
                bootstrap_estimates_dict[var_idx][i, 1] = np.mean(bs_results.correction[var_idx])  # Average correction
                # Average estimate 
                original_estimator_values = bs_results.calculate_estimator_at_points_parallel(
                    self.model.exog, var_idx, n_jobs=4, backend='threading'
                )
                bootstrap_estimates_dict[var_idx][i, 2] = np.mean(original_estimator_values)

                bootstrap_estimates_dict[var_idx][i, 3] = bs_results.estimate_at_point(
                    original_X_mean, var_idx  # Use original average, not bootstrap average
                )
        
        self.bootstrap = True
        self.bootstrap_reps = bootstrap_reps
        self.bootstrap_estimates_dict = bootstrap_estimates_dict
        
        # Calculate bootstrap standard errors for each variable
        self.bootstrap_se_dict = {var_idx: np.std(bootstrap_estimates_dict[var_idx], axis=0) 
                                 for var_idx in var_indices}

    def summary(self):
        """
        Provide a summary of the results with corrected standard errors.
        
        Returns
        -------
        Summary
            Summary object containing results tables and statistics
        """
        from statsmodels.iolib.summary2 import Summary
        
        smry = Summary()
        
        # Add a title
        if self.model.elasticity:
            title = "Doubly Robust Elasticity Estimator Results"
        else:
            title = "Doubly Robust Semi-Elasticity Estimator Results"
            
        smry.add_title(title)
        
        # Add OLS results
        smry.add_text("OLS Results:")
        # Get the OLS summary table and convert to DataFrame
        ols_summary_table = self.ols_results.summary().tables[1]
        if hasattr(ols_summary_table, 'data'):
            # Extract data from the table
            ols_data = []
            for row in ols_summary_table.data[1:]:  # Skip header
                ols_data.append(row)
            ols_df = pd.DataFrame(ols_data, columns=ols_summary_table.data[0])
            smry.add_df(ols_df)
        else:
            smry.add_text(str(ols_summary_table))
        
        # Build a table of corrected estimates for each variable of interest
        data = []
        
        # Add rows for each variable of interest
        for i, var_idx in enumerate(self.interest):
            var_name = self.model.exog_names[var_idx]
            row = {}
            row["Variable"] = var_name
            
            # Add OLS estimate
            if self.all_variables:
                beta_hat = self.beta_hat[var_idx]
            else:
                beta_hat = self.beta_hat[i]
            row["OLS Estimate"] = f"{beta_hat:.4f}"
            
            # Add average correction
            avg_corr = np.mean(self.correction[var_idx])
            row["Average Correction"] = f"{avg_corr:.4f}"
            
            # Add average estimate
            avg_est = np.mean(self.estimator_values[var_idx])
            row["Average Estimate"] = f"{avg_est:.4f}"
            
            # Add estimate at average
            est_at_avg = self.estimate_at_average(var_idx)
            row["Estimate at Average"] = f"{est_at_avg:.4f}"
            
            # Add standard errors and CI based on bootstrap or asymptotic
            if self.bootstrap:
                if hasattr(self, 'bootstrap_se_dict'):
                    se = self.bootstrap_se_dict[var_idx][3]  # Index 3 is for 'Estimate at Average'
                    row["Bootstrap SE"] = f"{se:.4f}"
                    row["95% CI Lower"] = f"{est_at_avg - 1.96 * se:.4f}"
                    row["95% CI Upper"] = f"{est_at_avg + 1.96 * se:.4f}"
                else:
                    row["Bootstrap SE"] = "N/A"
                    row["95% CI Lower"] = "N/A"
                    row["95% CI Upper"] = "N/A"
            else:
                try:
                    se = self.asymptotic_standard_error_at_average(var_idx)
                    row["Asymptotic SE"] = f"{se:.4f}"
                    row["95% CI Lower"] = f"{est_at_avg - 1.96 * se:.4f}"
                    row["95% CI Upper"] = f"{est_at_avg + 1.96 * se:.4f}"
                except Exception as e:
                    warnings.warn(f"Could not compute asymptotic SE: {e}")
                    row["Asymptotic SE"] = "N/A"
                    row["95% CI Lower"] = "N/A"
                    row["95% CI Upper"] = "N/A"
            
            data.append(row)
        
        # Create DataFrame from the data dictionary
        results_df = pd.DataFrame(data)
        results_df = results_df.set_index("Variable")
        
        smry.add_text("\nDoubly Robust Estimates:")
        smry.add_df(results_df)
        
        # Add information about the nonparametric component
        smry.add_text(f"\nNonparametric Component: {self.model.estimator_type}")
        if self.model.estimator_type == 'kernel' and hasattr(self.nonparam_model, 'bandwidth'):
            smry.add_text(f"Bandwidth: {self.nonparam_model.bandwidth}")
        
        # Add information about bootstrap if used
        if self.bootstrap:
            smry.add_text(f"\nBootstrap: {self.bootstrap_reps} replications")
        
        # Add information about asymptotic variance
        smry.add_text(f"\nAsymptotic Variance: Corrected multivariate implementation")
        smry.add_text(f"Kernel type: {self.kernel_type}")
        smry.add_text(f"Density estimator: {self.density_bandwidth_method}")
        
        return smry

    # Keep all other existing methods unchanged
    def estimate_at_average(self, variable_idx=None):
        """Calculate the estimator at the average values of regressors."""
        X_mean = np.mean(self.model.exog, axis=0).reshape(1, -1) 
        return self.estimate_at_point(X_mean, variable_idx) 
    
    def average_estimate(self, variable_idx=None):
        """Calculate the average estimate for values of regressors."""
        if isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        elif variable_idx is None:
            variable_idx = self.interest[0]

        return np.mean(self.estimator_values[variable_idx])
    
    def estimate_at_point(self, X_points, variable_idx=None):
        """
        Calculate the estimator at specific point(s) X_points.
        X_points can be a single point (1D or 2D array) or multiple points (2D array).
        Returns an array of estimates.
        """
        if not isinstance(X_points, np.ndarray):
            X_points = np.array(X_points)
        if X_points.ndim == 1:
            X_points = X_points.reshape(1, -1)
        
        num_points = X_points.shape[0]
        estimates = np.zeros(num_points)

        # Handle variable selection
        if variable_idx is None:
            if not self.interest: # Should not happen if model is fit
                raise ValueError("No variable of interest specified or found.")
            variable_idx = self.interest[0] # Default to the first variable in the interest list
        elif isinstance(variable_idx, str):
            if variable_idx not in self.model.exog_names:
                raise ValueError(f"Variable name '{variable_idx}' not in exog_names: {self.model.exog_names}")
            variable_idx = self.model.exog_names.index(variable_idx)
        # variable_idx is now an integer index for self.model.exog

        # Get OLS beta and log_x for the specific variable_idx
        current_ols_beta = self.ols_results.params[variable_idx]
        try:
            # Find position of variable_idx in the model's interest list to get corresponding log_x
            idx_in_model_interest_list = self.model.interest.index(variable_idx)
            current_log_x_bool = self.model.log_x[idx_in_model_interest_list]
        except ValueError:
            # Should not happen if variable_idx is from self.interest,
            # but as a fallback if variable_idx was passed freely:
            warnings.warn(f"Variable index {variable_idx} not in model.interest. Defaulting log_x to False for it.")
            current_log_x_bool = False


        if variable_idx in self.model.binary_vars:
            X_0 = X_points.copy()
            X_1 = X_points.copy()
            X_0[:, variable_idx] = 0
            X_1[:, variable_idx] = 1
            
            m_0 = self.nonparam_model.predict(X_0) # Expected 1D array of size num_points
            m_1 = self.nonparam_model.predict(X_1) # Expected 1D array
            
            m_0[m_0 == 0] = 1e-9 # Avoid division by zero
            estimates = np.exp(current_ols_beta) * (m_1 / m_0) - 1
        else:
            m_at_points = self.nonparam_model.predict(X_points) # Expected 1D array
            m_prime_at_points = self.nonparam_model.derivative(X_points, variable_idx) # Expected 1D array

            m_at_points[m_at_points == 0] = 1e-9 # Avoid division by zero
            correction = m_prime_at_points / m_at_points
            
            if self.model.elasticity and current_log_x_bool:
                estimates = current_ols_beta + correction
            elif not self.model.elasticity and not current_log_x_bool:
                estimates = current_ols_beta + correction
            elif self.model.elasticity and not current_log_x_bool:
                estimates = current_ols_beta + correction * X_points[:, variable_idx]
            else:  # not self.model.elasticity and current_log_x_bool
                x_val_col = X_points[:, variable_idx].copy()
                x_val_col[x_val_col == 0] = 1e-9
                estimates = current_ols_beta + correction / x_val_col
        
        return estimates if num_points > 1 else estimates[0] # Return scalar if single point input


    def calculate_estimator_at_points_parallel(self, X_points, variable_idx=None, n_jobs=-1, backend='threading'):
        """Calculate estimator values with parallel processing (pickle-safe)."""
        import joblib # Moved import here
        
        if not isinstance(X_points, np.ndarray):
            X_points = np.array(X_points)
        if X_points.ndim == 1: # If a single point is passed
             X_points = X_points.reshape(1, -1)

        if backend == 'threading' and n_jobs == 1: # No benefit from parallel overhead for 1 job
            return self.estimate_at_point(X_points, variable_idx)
        
        # If estimate_at_point is now fast and vectorized, this parallel map might be slower due to overhead
        # unless X_points is huge and estimate_at_point still has bottlenecks (e.g. ordinal derivative loops)
        # For simplicity, let's assume it processes point-by-point for parallel to use the existing structure
        
        if backend == 'threading':
            # For threading, we can still use the new estimate_at_point but apply it per row
            # if we want to keep the existing joblib structure.
            # However, if estimate_at_point is vectorized, just call it once.
            # Let's assume for now the user of parallel wants to parallelize over rows.
            estimator_values = joblib.Parallel(n_jobs=n_jobs, backend='threading')(
                joblib.delayed(lambda x_row: self.estimate_at_point(x_row.reshape(1, -1), variable_idx))(x_point_row) 
                for x_point_row in X_points
            )
        else: # multiprocessing
            # _estimate_single_point_worker expects self (results_obj) to have estimate_at_point
            # that takes a single point.
            args = [(x_point_row, self, variable_idx) for x_point_row in X_points]
            estimator_values = joblib.Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                joblib.delayed(_estimate_single_point_worker)(arg) for arg in args
            )
        
        return np.array(estimator_values).flatten() # Ensure it's a flat array like other methods
    

    
    def calculate_estimator_at_points(self, X_points, variable_idx=None):
        """
        Calculate estimator values at specific X points.
        This method is now a direct wrapper for the vectorized estimate_at_point.
        """
        return self.estimate_at_point(X_points, variable_idx)

    def predict(self, exog=None):
        """Predict using the model."""
        if exog is None:
            exog = self.model.exog
            
        # Get the OLS predictions on the log scale
        log_pred = self.ols_results.predict(exog)
        
        # Get the correction terms
        m_hat = self.nonparam_model.predict(exog)
        
        # Return transformed predictions (exponentiated and corrected)
        return np.exp(log_pred) * m_hat

    def plot_distribution(self, variable_idx=None):
        """Plot the distribution of estimator values."""
        # Convert variable name to index if needed
        if isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        elif variable_idx is None:
            variable_idx = self.interest[0]
            
        # Get the position of this variable in the interest list
        if self.all_variables:
            i = list(self.interest).index(variable_idx)
            beta_hat = self.beta_hat[variable_idx]
            var_name = self.model.exog_names[variable_idx]
        else:
            i = self.interest.index(variable_idx)
            beta_hat = self.beta_hat[i]
            var_name = self.model.exog_names[variable_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.kdeplot(self.estimator_values[variable_idx], ax=ax)
        ax.axvline(x=beta_hat, color='red', linestyle='--', label='OLS Estimate')
        ax.axvline(x=np.mean(self.estimator_values[variable_idx]), color='blue', linestyle='--', 
                  label='Doubly Robust Average')
        ax.axvline(x=self.estimate_at_average(variable_idx), color='green', linestyle='--', 
                  label='Estimate at Average')
        
        if self.model.elasticity:
            ax.set_title(f'Distribution of Elasticity Estimates for {var_name}')
            ax.set_xlabel('Elasticity')
        else:
            ax.set_title(f'Distribution of Semi-Elasticity Estimates for {var_name}')
            ax.set_xlabel('Semi-Elasticity')
        
        ax.set_ylabel('Density')
        ax.legend()
        
        return fig
    
    def plot_correction(self, variable_idx=None):
        """Plot the correction term against the regressor of interest."""
        # Convert variable name to index if needed
        if isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        elif variable_idx is None:
            variable_idx = self.interest[0]
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_values = self.model.exog[:, variable_idx]
        
        sns.scatterplot(x=x_values, y=self.correction[variable_idx], ax=ax, alpha=0.5)
        
        # Add a smoothed line
        sns.regplot(x=x_values, y=self.correction[variable_idx], ax=ax, scatter=False, 
                   lowess=True, line_kws={'color': 'red'})
        
        ax.set_title(f'Correction Term vs {self.model.exog_names[variable_idx]}')
        ax.set_xlabel(f'Regressor: {self.model.exog_names[variable_idx]}')
        ax.set_ylabel('Correction Term')
        
        return fig
    
    def test_ppml(self):
        """Test the consistency of PPML estimation."""
        ppml_mod = sm.GLM(self.model.endog, self.model.exog, 
                          family=sm.families.Poisson()).fit(cov_type='HC3')
        # Assuming AssumptionTest is defined as imported
        from .ppml_consistency import AssumptionTest
        return AssumptionTest(ppml_mod).test_direct()


class RegressionModel:
    """Base class for nonparametric regression models.
    
    This abstract class defines the interface for all nonparametric regression models
    used to estimate m(x) = E[exp(u)|X=x] and its gradient.
    """
    
    def __init__(self, binary_vars, ordinal_vars):
        """Initialize the regression model."""
        self.binary_vars = binary_vars
        self.ordinal_vars = ordinal_vars
    
    def predict(self, X):
        """Predict the expected value m(x) for given data."""
        pass
    
    def derivative(self, X, index):
        """Calculate the derivative of m(x) with respect to a specific regressor."""
        pass


class KernelRegressionModel(RegressionModel):
    """Kernel regression model for nonparametric estimation.
    
    This class implements a local linear kernel regression model for estimating
    m(x) = E[exp(u)|X=x] and its gradient.
    """
    
    def __init__(self, model, binary_vars, ordinal_vars, scaling_factor=1.0):
        """Initialize the kernel regression model.
        
        Parameters
        ----------
        model : KernelReg
            The fitted kernel regression model
        binary_vars : array-like
            Indices of binary variables
        ordinal_vars : array-like
            Indices of ordinal variables
        scaling_factor : float, optional
            Scaling factor used to normalize the exponentiated residuals (min(exp_residuals))
        """
        super(KernelRegressionModel, self).__init__(binary_vars, ordinal_vars)
        self.model = model
        self.bandwidth = np.mean(model.bw)  # For simplicity, use mean bandwidth
        self.scaling_factor = scaling_factor
    
    def predict(self, X):
        """Predict the expected value m(x) for given data."""
        # Return predictions scaled back to original scale
        return self.model.fit(X)[0] * self.scaling_factor

    def derivative(self, X, index):
        """Calculate the derivative of m(x) with respect to a specific regressor."""
        # For binary variables, use difference instead of derivative
        if index in self.binary_vars:
            X_0 = X.copy()
            X_1 = X.copy()
            X_0[:, index] = 0
            X_1[:, index] = 1
            m_0 = self.predict(X_0)
            m_1 = self.predict(X_1)
            return m_1 - m_0
            
        # For ordinal variables, use finite differences
        elif index in self.ordinal_vars:
            unique_values = np.sort(np.unique(X[:, index]))
            
            # If there's only one unique value, derivative is zero
            if len(unique_values) <= 1:
                return np.zeros(len(X))
                
            if len(X) == 1:  # If X is a single point
                current_value = X[0, index]
                idx = np.searchsorted(unique_values, current_value)
                
                # Handle edge cases
                if len(unique_values) == 1:
                    return np.array([0.0])
                elif idx == 0:
                    if len(unique_values) > 1:
                        next_value = unique_values[1]
                        X_next = X.copy()
                        X_next[0, index] = next_value
                        return (self.predict(X_next) - self.predict(X)) / (next_value - current_value)
                    else:
                        return np.array([0.0])
                elif idx == len(unique_values) - 1:
                    prev_value = unique_values[-2]
                    X_prev = X.copy()
                    X_prev[0, index] = prev_value
                    return (self.predict(X) - self.predict(X_prev)) / (current_value - prev_value)
                else:
                    prev_value = unique_values[idx-1]
                    next_value = unique_values[idx+1]
                    X_prev = X.copy()
                    X_next = X.copy()
                    X_prev[0, index] = prev_value
                    X_next[0, index] = next_value
                    return (self.predict(X_next) - self.predict(X_prev)) / (next_value - prev_value)
            else:
                # For multiple points, calculate derivative at each point
                derivatives = np.zeros(len(X))
                for i in range(len(X)):
                    x_i = X[i:i+1]
                    derivatives[i] = self.derivative(x_i, index)[0]
                return derivatives
        
        # For continuous variables, use the marginal effects from KernelReg.fit
        else:
            # Get both mean and marginal effects from fit
            try:
                mean, mfx = self.model.fit(X)
            except Exception as e:
                # Fallback to numerical differentiation if kernel regression fails
                print("Falling back to numerical differentiation in kernel regression.")
                epsilon = 1e-3
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[:, index] += epsilon
                X_minus[:, index] -= epsilon
                return (self.predict(X_plus) - self.predict(X_minus)) / (2 * epsilon)
            
            # Handle different formats of mfx that might be returned in different versions
            if isinstance(mfx, list):
                # If mfx is a list of arrays (one per variable), extract the one we want
                if len(mfx) > index:
                    return mfx[index] * self.scaling_factor
                else:
                    # Fallback to numerical differentiation if mfx format is unexpected
                    epsilon = 1e-3
                    X_plus = X.copy()
                    X_minus = X.copy()
                    X_plus[:, index] += epsilon
                    X_minus[:, index] -= epsilon
                    return (self.predict(X_plus) - self.predict(X_minus)) / (2 * epsilon)
            elif isinstance(mfx, np.ndarray):
                # If mfx is a 2D array (n_obs, n_vars) or (n_vars, n_obs)
                if len(mfx.shape) == 2:
                    if mfx.shape[0] == X.shape[0]:  # (n_obs, n_vars)
                        return mfx[:, index] * self.scaling_factor
                    elif mfx.shape[1] == X.shape[0]:  # (n_vars, n_obs)
                        return mfx[index, :] * self.scaling_factor
                    else:
                        # Handle case where dimensions don't match
                        return np.full(X.shape[0], np.mean(mfx)) * self.scaling_factor
                else:
                    # If it's a 1D array, assume it's already the derivative we want
                    return mfx * self.scaling_factor
            else:
                # Fallback to numerical differentiation as a last resort
                epsilon = 1e-3
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[:, index] += epsilon
                X_minus[:, index] -= epsilon
                return (self.predict(X_plus) - self.predict(X_minus)) / (2 * epsilon)
    


class NNRegressionModel(RegressionModel):
    """Neural network regression model for nonparametric estimation.
    
    This class implements a neural network model for estimating
    m(x) = E[exp(u)|X=x] and its gradient.
    """
    
    def __init__(self, model, mean, std, scaling_factor, binary_vars, ordinal_vars):
        """Initialize the neural network regression model.
        
        Parameters
        ----------
        model : tensorflow.keras.Model
            The fitted neural network model
        mean : float
            Mean used for standardizing the response variable
        std : float
            Standard deviation used for standardizing the response variable
        scaling_factor : float
            Scaling factor used to normalize the exponentiated residuals (min(exp_residuals))
        binary_vars : array-like
            Indices of binary variables
        ordinal_vars : array-like
            Indices of ordinal variables
        """
        super(NNRegressionModel, self).__init__(binary_vars, ordinal_vars)
        self.model = model
        self.mean = mean
        self.std = std
        self.scaling_factor = scaling_factor
        self.bandwidth = 0.1  # Placeholder, not actually used
        
        # Import tensorflow only when needed
        import tensorflow as tf
        self.tf = tf
    
    def predict(self, X):
        """Predict the expected value m(x) for given data."""
        X_tensor = self.tf.convert_to_tensor(X, dtype=self.tf.float32)
        predictions = self.model.predict(X_tensor)
        # Unstandardize and unscale to get back to original scale
        return (predictions.reshape(-1) * self.std + self.mean) * self.scaling_factor
    
    def derivative(self, X, index):
        """Calculate the derivative of m(x) with respect to a specific regressor."""
        # For binary variables, use difference instead of derivative
        if index in self.binary_vars:
            X_0 = X.copy()
            X_1 = X.copy()
            X_0[:, index] = 0
            X_1[:, index] = 1
            m_0 = self.predict(X_0)
            m_1 = self.predict(X_1)
            return m_1 - m_0
            
        # For ordinal variables, use finite differences
        elif index in self.ordinal_vars:
            unique_values = np.sort(np.unique(X[:, index]))
            if len(X) == 1:  # If X is a single point
                current_value = X[0, index]
                idx = np.searchsorted(unique_values, current_value)
                if idx == 0:
                    next_value = unique_values[1]
                    X_next = X.copy()
                    X_next[0, index] = next_value
                    return (self.predict(X_next) - self.predict(X)) / (next_value - current_value)
                elif idx == len(unique_values) - 1:
                    prev_value = unique_values[-2]
                    X_prev = X.copy()
                    X_prev[0, index] = prev_value
                    return (self.predict(X) - self.predict(X_prev)) / (current_value - prev_value)
                else:
                    prev_value = unique_values[idx-1]
                    next_value = unique_values[idx+1]
                    X_prev = X.copy()
                    X_next = X.copy()
                    X_prev[0, index] = prev_value
                    X_next[0, index] = next_value
                    return (self.predict(X_next) - self.predict(X_prev)) / (next_value - prev_value)
            else:
                # For multiple points, calculate derivative at each point
                derivatives = np.zeros(len(X))
                for i in range(len(X)):
                    x_i = X[i:i+1]
                    derivatives[i] = self.derivative(x_i, index)[0]
                return derivatives
        
        # For continuous variables, use automatic differentiation
        else:
            X_tensor = self.tf.convert_to_tensor(X, dtype=self.tf.float32)
            X_var = self.tf.Variable(X_tensor)
            
            with self.tf.GradientTape() as tape:
                tape.watch(X_var)
                # Account for both standardization and scaling
                predictions = self.model(X_var) * self.std * self.scaling_factor + self.mean * self.scaling_factor
                
            gradients = tape.gradient(predictions, X_var)
            return gradients[:, index].numpy()


class BinaryRegressionModel(RegressionModel):
    """Binary regression model for nonparametric estimation.
    
    This class implements a simple model for when the regressor of interest is binary.
    It estimates m(x) = E[exp(u)|X=x] separately for each value of the binary variable.
    """
    
    def __init__(self, eu_0, eu_1, binary_index, scaling_factor, binary_vars, ordinal_vars):
        """Initialize the binary regression model.
        
        Parameters
        ----------
        eu_0 : float
            Expected value of normalized exp(u) when binary variable = 0
        eu_1 : float
            Expected value of normalized exp(u) when binary variable = 1
        binary_index : int
            Index of the binary variable of interest
        scaling_factor : float
            Scaling factor used to normalize the exponentiated residuals (min(exp_residuals))
        binary_vars : array-like
            Indices of binary variables
        ordinal_vars : array-like
            Indices of ordinal variables
        """
        super(BinaryRegressionModel, self).__init__(binary_vars, ordinal_vars)
        self.eu_0 = eu_0
        self.eu_1 = eu_1
        self.binary_index = binary_index
        self.scaling_factor = scaling_factor
        self.bandwidth = None  # Not applicable for binary model
    
    def predict(self, X):
        """Predict the expected value m(x) for given data."""
        # Return predictions scaled back to original scale
        return np.where(X[:, self.binary_index] == 0, 
                       self.eu_0 * self.scaling_factor, 
                       self.eu_1 * self.scaling_factor)
    
    def derivative(self, X, index):
        """Calculate the 'derivative' (difference) for the binary variable."""
        if index != self.binary_index:
            raise ValueError("For binary model, derivative is only defined for the binary variable of interest.")
        return np.ones(len(X)) * (self.eu_1 - self.eu_0) * self.scaling_factor



class MultiBinaryRegressionModel(RegressionModel):
    """Model for multiple binary variables.
    
    This class implements a model for when there are multiple binary variables of interest.
    It estimates m(x) = E[exp(u)|X=x] separately for each value of each binary variable.
    """
    
    def __init__(self, binary_models, scaling_factor, binary_vars, ordinal_vars):
        """Initialize the multi-binary regression model.
        
        Parameters
        ----------
        binary_models : dict
            Dictionary mapping variable indices to tuples of (eu_0, eu_1)
        scaling_factor : float
            Scaling factor used to normalize the exponentiated residuals
        binary_vars : array-like
            Indices of binary variables
        ordinal_vars : array-like
            Indices of ordinal variables
        """
        super(MultiBinaryRegressionModel, self).__init__(binary_vars, ordinal_vars)
        self.binary_models = binary_models
        self.scaling_factor = scaling_factor
        self.bandwidth = None  # Not applicable for binary model
    
    def predict(self, X):
        """Predict the expected value m(x) for given data."""
        # If no binary models, return scaling factor
        if not self.binary_models:
            return np.ones(len(X)) * self.scaling_factor
            
        # Start with array of ones as base prediction
        predictions = np.ones(len(X))
        
        # For the first binary variable model, use it as the base
        first_var_idx = list(self.binary_models.keys())[0]
        eu_0, eu_1 = self.binary_models[first_var_idx]
        x_binary = X[:, first_var_idx]
        predictions = np.where(x_binary == 0, eu_0, eu_1)
        
        # Scale back to original scale
        return predictions * self.scaling_factor
    
    def derivative(self, X, index):
        """Calculate the 'derivative' (difference) for a binary variable."""
        if index not in self.binary_models:
            # If the variable is not in our binary models, return zeros
            return np.zeros(len(X))
            
        eu_0, eu_1 = self.binary_models[index]
        
        # Calculate the derivative as the difference in predictions
        return np.ones(len(X)) * (eu_1 - eu_0) * self.scaling_factor



class OLSRegressionModel(RegressionModel):
    """OLS regression model with polynomial features for nonparametric estimation.
    
    This class implements a polynomial OLS regression model for estimating
    m(x) = E[exp(u)|X=x] and its gradient using numerical differentiation.
    """
    
    def __init__(self, model, poly, scaling_factor, binary_vars, ordinal_vars):
        """Initialize the OLS regression model.
        
        Parameters
        ----------
        model : statsmodels.regression.linear_model.RegressionResults
            The fitted OLS model on polynomial features
        poly : sklearn.preprocessing.PolynomialFeatures
            The polynomial features transformer
        scaling_factor : float
            Scaling factor used to normalize the exponentiated residuals (min(exp_residuals))
        binary_vars : array-like
            Indices of binary variables
        ordinal_vars : array-like
            Indices of ordinal variables
        """
        super(OLSRegressionModel, self).__init__(binary_vars, ordinal_vars)
        self.model = model
        self.poly = poly
        self.scaling_factor = scaling_factor
        self.bandwidth = None  # Not applicable for OLS model
    
    def predict(self, X):
        """Predict the expected value m(x) for given data."""
        X_poly = self.poly.transform(X)
        # Return predictions scaled back to original scale
        return self.model.predict(X_poly) * self.scaling_factor
    
    def derivative(self, X, index):
        """Calculate the derivative of m(x) with respect to a specific regressor."""
        # For binary variables, use difference instead of derivative
        if index in self.binary_vars:
            X_0 = X.copy()
            X_1 = X.copy()
            X_0[:, index] = 0
            X_1[:, index] = 1
            m_0 = self.predict(X_0)
            m_1 = self.predict(X_1)
            return m_1 - m_0
            
        # For ordinal variables, use finite differences
        elif index in self.ordinal_vars:
            unique_values = np.sort(np.unique(X[:, index]))
            
            # If there's only one unique value, derivative is zero
            if len(unique_values) <= 1:
                return np.zeros(len(X))
                
            if len(X) == 1:  # If X is a single point
                current_value = X[0, index]
                idx = np.searchsorted(unique_values, current_value)
                
                # Handle edge cases
                if len(unique_values) == 1:
                    return np.array([0.0])
                elif idx == 0:
                    if len(unique_values) > 1:
                        next_value = unique_values[1]
                        X_next = X.copy()
                        X_next[0, index] = next_value
                        return (self.predict(X_next) - self.predict(X)) / (next_value - current_value)
                    else:
                        return np.array([0.0])
                elif idx == len(unique_values) - 1:
                    prev_value = unique_values[-2]
                    X_prev = X.copy()
                    X_prev[0, index] = prev_value
                    return (self.predict(X) - self.predict(X_prev)) / (current_value - prev_value)
                else:
                    prev_value = unique_values[idx-1]
                    next_value = unique_values[idx+1]
                    X_prev = X.copy()
                    X_next = X.copy()
                    X_prev[0, index] = prev_value
                    X_next[0, index] = next_value
                    return (self.predict(X_next) - self.predict(X_prev)) / (next_value - prev_value)
            else:
                # For multiple points, calculate derivative at each point
                derivatives = np.zeros(len(X))
                for i in range(len(X)):
                    x_i = X[i:i+1]
                    derivatives[i] = self.derivative(x_i, index)[0]
                return derivatives
        
        # For continuous variables, use numerical differentiation
        else:
            epsilon = 1e-3
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[:, index] += epsilon
            X_minus[:, index] -= epsilon
            return (self.predict(X_plus) - self.predict(X_minus)) / (2 * epsilon)



class NNConditionalVarianceModel:
    """
    Neural network model for estimating conditional variance σ²_w(x) = Var(e^u|X=x).
    
    This class implements a neural network to estimate E[w²|X] where w = e^u,
    which is then used to compute the conditional variance as E[w²|X] - (E[w|X])².
    
    Parameters
    ----------
    model : tensorflow.keras.Model
        Fitted neural network model for E[w²|X]
    mean : float
        Mean used for standardizing the squared residuals
    std : float
        Standard deviation used for standardizing the squared residuals
    scaling_factor : float
        Scaling factor used to normalize the exponentiated residuals
    binary_vars : array-like
        Indices of binary variables
    ordinal_vars : array-like
        Indices of ordinal variables
        
    Attributes
    ----------
    model : tensorflow.keras.Model
        The fitted neural network
    mean : float
        Normalization mean for squared residuals
    std : float
        Normalization standard deviation for squared residuals
    scaling_factor : float
        Original scaling factor
    tf : module
        TensorFlow module reference
    """
    
        
    def __init__(self, model, mean, std, scaling_factor, binary_vars, ordinal_vars):
        """Initialize the neural network conditional variance model."""
        self.model = model
        self.mean = mean
        self.std = std
        self.scaling_factor = scaling_factor
        self.binary_vars = binary_vars
        self.ordinal_vars = ordinal_vars
        
        # Import tensorflow only when needed
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            raise ImportError("TensorFlow is required for neural network conditional variance.")
    
    def fit(self, X):
        """
        Predict E[w²|X] at given points using conventional .fit() interface.
        
        Parameters
        ----------
        X : array_like of shape (n, d)
            Input points for prediction
            
        Returns
        -------
        tuple
            (predictions, marginal_effects) where:
            - predictions: ndarray of shape (n,) with E[w²|X] values
            - marginal_effects: None (for compatibility with KernelReg)
            
        Notes
        -----
        Returns the same format as statsmodels.nonparametric.kernel_regression.KernelReg.fit()
        for seamless integration with existing conditional variance code.
        """
        X_tensor = self.tf.convert_to_tensor(X, dtype=self.tf.float32)
        
        # Get predictions from neural network (in standardized scale)
        predictions = self.model.predict(X_tensor, verbose=0)
        
        # Denormalize: reverse standardization from training
        unstandardized = predictions.reshape(-1) * self.std + self.mean
        
        # Unscale: reverse scaling for numerical stability
        unscaled = unstandardized * self.scaling_factor
        
        # Ensure non-negative predictions (variance must be non-negative)
        unscaled = np.maximum(unscaled, 0.0)
        
        # Return in KernelReg.fit() format: (predictions, marginal_effects)
        return (unscaled, None)


# Alias names for easier use
DRE = DoublyRobustElasticityEstimator
DREER = DoublyRobustElasticityEstimatorResults


# Example usage
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Generate some sample data
    np.random.seed(6699)
    n = 200
    x1 = np.random.normal(5, 1, n)
    #x2 = (np.random.uniform(0, 1, n) > 0.5).astype(int)  # Binary variable
    #x3 = np.random.choice(np.arange(1, 6), n)  # Ordinal variable (1-5)
    
    # Create a log-linear model with heteroskedasticity
    log_y = 0.5 + 1.2 * x1 + \
        np.random.normal(0, 1 + 0.1 * x1 ** 2 , n)
        # 0.8 * x2 + 0.3 * x3 + 

    y = np.exp(log_y)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'y': y,
        'x1': x1,
        #'x2': x2,
        #'x3': x3
    })
    
    # Fit standard OLS model
    import statsmodels.api as sm
    X = sm.add_constant(df[[
        'x1', 
        #'x2', 'x3'
    ]])
    ols_mod = sm.OLS(np.log(df['y']), X).fit()
    print("OLS results:")
    print(ols_mod.summary())


    # Fit PPML
    ppml_mod = sm.GLM(df['y'], X, family=sm.families.Poisson()).fit()
    print("PPML results:")
    print(ppml_mod.summary())
    
    # Option 1: Fit with a specific variable of interest
    print("\n1. Fitting with a specific variable of interest (x1):")
    dre1 = DRE(
        df['y'],
        df[['x1', 
            #'x2', 'x3'
            ]],
        interest='x1',
        estimator_type='nn',
        elasticity=False,
        density_estimator='kernel',
        kernel_params={'degree':2},
    )
    dre_results1 = dre1.fit()
    print(dre_results1.summary())
    #
    ## Option 2: Fit with multiple variables of interest
    #print("\n2. Fitting with multiple variables of interest (x1 and x2):")
    #dre2 = DRE(
    #    df['y'],
    #    df[['x1', 'x2', 'x3']],
    #    interest=['x1', 'x2'],
    #    log_x=[False, False],
    #    estimator_type='kernel',
    #    elasticity=False,
    #    density_estimator='kernel'
    #)
    #dre_results2 = dre2.fit()
    #print(dre_results2.summary())
    #
    ## Option 3: Fit for all variables
    #print("\n3. Fitting for all variables:")
    #dre3 = DRE(
    #    df['y'],
    #    df[['x1', 'x2', 'x3']],
    #    interest=None,  # Calculate for all variables
    #    estimator_type='kernel',
    #    elasticity=False,
    #    density_estimator='kernel'
    #)
    #dre_results3 = dre3.fit()
    #print(dre_results3.summary())
    
    # Option 4: Fit with bootstrap
    print("\n4. Fitting with bootstrap standard errors:")
    dre4 = DRE(
        df['y'],
        df[['x1', 
            #'x2', 'x3'
            ]],
        interest='x1',
        estimator_type='nn',
        elasticity=False,
        density_estimator='kernel',
        kernel_params={'bw':'normal_reference', 'degree':2}
    )
    # Reduce bootstrap_reps for this example
    dre_results4 = dre4.fit(bootstrap=True, bootstrap_reps=10, bootstrap_method='pairs')
    print(dre_results4.summary())
    
    # Plot distribution of estimates for a specific variable
    fig = dre_results1.plot_distribution()
    plt.show()
    
    # Plot correction term for a specific variable
    fig = dre_results1.plot_correction('x1')
    plt.show()
    
    # If bootstrap was used, plot bootstrap distribution
    if hasattr(dre_results4, 'bootstrap') and dre_results4.bootstrap:
        fig, ax = dre_results4.plot_bootstrap_distribution('estimate_at_average')
        plt.show()
        
        # Get bootstrap confidence intervals
        ci = dre_results4.bootstrap_confidence_interval(method='percentile')
        print("\nBootstrap Confidence Intervals (95%):")
        for key, value in ci.items():
            print(f"{key}: [{value[0]:.4f}, {value[1]:.4f}]")

