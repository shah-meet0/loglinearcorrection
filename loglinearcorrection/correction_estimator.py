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


class DoublyRobustElasticityEstimatorResults(Results):
    """Results from the DoublyRobustElasticityEstimator.
    
    This class stores and presents results from the doubly robust elasticity estimator,
    providing methods to access various statistics, visualizations, and tests.
    
    Parameters
    ----------
    model : DoublyRobustElasticityEstimator
        The estimator model
    ols_results : RegressionResults
        Results from the initial OLS model
    nonparam_model : RegressionModel
        The fitted nonparametric model
    exp_residuals : array-like
        Exponentiated residuals from the parametric model
    density_model : NNDensityModel or None, optional
        The fitted density model for high-dimensional X
    bootstrap : bool, optional
        Whether bootstrap standard errors have been computed
    bootstrap_reps : int, optional
        Number of bootstrap replications
    bootstrap_estimates : array-like, optional
        Bootstrap estimates
    """
    
    def __init__(self, model, ols_results, nonparam_model, exp_residuals, 
                 density_model=None, bootstrap=False, bootstrap_reps=None, 
                 bootstrap_estimates=None):
        """Initialize the results object."""
        self.model = model
        self.ols_results = ols_results
        self.nonparam_model = nonparam_model
        self.exp_residuals = exp_residuals
        self.interest = model.interest
        self.all_variables = model.all_variables
        self.density_model = density_model
        
        # Bootstrap-related attributes
        self.bootstrap = bootstrap
        self.bootstrap_reps = bootstrap_reps
        self.bootstrap_estimates = bootstrap_estimates
        
        # Extract the OLS estimate of the parameter(s) of interest
        if self.all_variables:
            self.beta_hat = ols_results.params
        else:
            self.beta_hat = [ols_results.params[idx] for idx in self.interest]
        
        # Compute correction terms for all data points
        self._compute_corrections()
        
        # Initialize the base Results class
        super(DoublyRobustElasticityEstimatorResults, self).__init__(model, ols_results.params)
        
    def _compute_corrections(self):
        """Compute correction terms for all data points in the sample."""
        # TODO: Fix for percentage changes case so its e^beta1 * m1/m0 - 1
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
    
    def _bootstrap_standard_errors(self, bootstrap_reps, bootstrap_method, weights, kwargs):
        """Compute bootstrap standard errors.
        
        Parameters
        ----------
        bootstrap_reps : int
            Number of bootstrap replications
        bootstrap_method : str
            Bootstrap method to use: 'pairs' or 'residuals'
        weights : array-like or None
            Weights for WLS if provided
        kwargs : dict
            Additional arguments for the OLS fit method
            
        Returns
        -------
        None
            Results are stored in bootstrap_estimates attribute
        """
        n = len(self.model.endog)
        
        # Initialize storage for bootstrap estimates
        # For each variable of interest, we track 4 estimates:
        # 1. OLS estimate
        # 2. Average correction
        # 3. Average estimate
        # 4. Estimate at average X
        if self.all_variables:
            var_indices = self.interest
        else:
            var_indices = self.interest
            
        bootstrap_estimates_dict = {var_idx: np.zeros((bootstrap_reps, 4)) 
                                   for var_idx in var_indices}
        
        for i in range(bootstrap_reps):
            if i > 0 and i % 100 == 0:
                print(f"  Completed {i} bootstrap replications")
                
            if bootstrap_method == 'pairs':
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
                
            elif bootstrap_method == 'residuals': #DO NOT USE - WILL BE REMOVED 
                # Get fitted values from OLS model
                fitted_values = np.exp(self.ols_results.fittedvalues)
                
                # Resample residuals
                residual_indices = np.random.choice(n, size=n, replace=True)
                resampled_residuals = self.exp_residuals[residual_indices]
                
                # Create new outcome by multiplying fitted values with resampled residuals
                bs_endog = fitted_values * resampled_residuals
                
                # Use original exog
                bs_exog = self.model.exog
                
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
            
            else:
                raise ValueError(f"Bootstrap method '{bootstrap_method}' not supported. "
                                f"Use 'pairs' or 'residuals'.")
            
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
                # TODO: make sure the logic here allows non parallel processing too somehow
                original_estimator_values = bs_results.calculate_estimator_at_points_parallel(
                    self.model.exog, var_idx, n_jobs=4, backend='threading'
                )
                bootstrap_estimates_dict[var_idx][i, 2] = np.mean(original_estimator_values)


                bootstrap_estimates_dict[var_idx][i, 3] = bs_results.estimate_at_point(
                    original_X_mean, var_idx  # Use original average, not bootstrap average
                )

                print(f"Bootstrap estimate at average: {bootstrap_estimates_dict[var_idx][i,3]}")
                print(f"Bootstrap average elasticity estimate: {bootstrap_estimates_dict[var_idx][i,2]}")
        
        self.bootstrap = True
        self.bootstrap_reps = bootstrap_reps
        self.bootstrap_estimates_dict = bootstrap_estimates_dict
        
        # Calculate bootstrap standard errors for each variable
        self.bootstrap_se_dict = {var_idx: np.std(bootstrap_estimates_dict[var_idx], axis=0) 
                                 for var_idx in var_indices}
    
    def summary(self):
        """Provide a summary of the results."""
        from statsmodels.iolib.summary2 import Summary
        import pandas as pd
        
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
                    X_mean = np.mean(self.model.exog, axis=0).reshape(1, -1)
                    se = np.sqrt(self._asymptotic_variance(X_mean, var_idx))
                    row["Asymptotic SE"] = f"{se:.4f}"
                    row["95% CI Lower"] = f"{est_at_avg - 1.96 * se:.4f}"
                    row["95% CI Upper"] = f"{est_at_avg + 1.96 * se:.4f}"
                except:
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
        
        # Add information about density estimator
        smry.add_text(f"\nDensity Estimator: {self.model.density_estimator}")
        
        return smry

    
    def estimate_at_average(self, variable_idx=None):
        """Calculate the estimator at the average values of regressors.
        
        Parameters
        ----------
        variable_idx : int or str, optional
            Index or name of the variable to calculate for. If None, uses the first
            variable of interest. Defaults to None.
            
        Returns
        -------
        float
            Estimator value at average X
        """
        # Convert variable name to index if needed
        if isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        elif variable_idx is None:
            variable_idx = self.interest[0]
            
        # Get the index of this variable in the interest list
        if self.all_variables:
            i = list(self.interest).index(variable_idx)
            beta_hat = self.beta_hat[variable_idx]
        else:
            i = self.interest.index(variable_idx)
            beta_hat = self.beta_hat[i]
            
        X_mean = np.mean(self.model.exog, axis=0).reshape(1, -1)
        m_at_mean = self.nonparam_model.predict(X_mean)[0]
        m_prime_at_mean = self.nonparam_model.derivative(X_mean, variable_idx)[0]
        correction = m_prime_at_mean / m_at_mean
        
        if self.model.elasticity and self.model.log_x[i]:
            return beta_hat + correction
        elif not self.model.elasticity and not self.model.log_x[i]:
            return beta_hat + correction
        elif self.model.elasticity and not self.model.log_x[i]:
            return beta_hat + correction * X_mean[0, variable_idx]
        else:  # not self.model.elasticity and self.model.log_x[i]
            return beta_hat + correction / X_mean[0, variable_idx]
    
    def average_estimate(self, variable_idx=None):
        """Calculate the average estimate for values of regressors.

        Parameters
        ----------
        variable_idx : int or str, optional
            Index or name of the variable to calculate for. If None, uses the first
            variable of interest. Defaults to None.

        Returns
        -------
        float
            Estimator value at average X
        """
        if isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        elif variable_idx is None:
            variable_idx = self.interest[0]

        return np.mean(self.estimator_values[variable_idx])
    
    def std_error_at_average(self):
        """Calculate the standard error of the estimator at average values."""
        X_mean = np.mean(self.model.exog, axis=0).reshape(1, -1)
        return np.sqrt(self._asymptotic_variance(X_mean))
    
    def _asymptotic_variance(self, x, variable_idx=None):
        """Calculate the asymptotic variance of the estimator at point x.
        
        Parameters
        ----------
        x : array-like
            Point at which to evaluate the asymptotic variance
        variable_idx : int or str, optional
            Index or name of the variable to calculate for. If None, uses the first
            variable of interest. Defaults to None.
            
        Returns
        -------
        float
            Asymptotic variance
        """
        # Convert variable name to index if needed
        if isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        elif variable_idx is None:
            variable_idx = self.interest[0]
            
        # Get sample size
        n = self.model.exog.shape[0]
        
        # Compute density at point x using the appropriate estimator
        f_x = self._kernel_density(x)
        
        # Compute variance of exponentiated residuals
        var_w = np.var(self.exp_residuals)
        
        # Get m(x) and m'(x) at point x
        m_x = self.nonparam_model.predict(x)[0]
        m_prime_x = self.nonparam_model.derivative(x, variable_idx)[0]
        
        # Get bandwidth
        if self.nonparam_model.bandwidth is not None:
            h = self.nonparam_model.bandwidth
        else:
            # Default bandwidth for non-kernel models
            h = 1.0 * n ** (-1/(4 + self.model.exog.shape[1]))
        
        # Dimension of x
        d = self.model.exog.shape[1]
        
        # Compute kernel constants for univariate case
        # For multivariate product kernels
        nu_0_K = 0.2821  # Default value for Gaussian kernel
        nu_2_K = 0.0856  # Default value for Gaussian kernel
        mu_2_K = 1.0     # Default value for Gaussian kernel
        
        # First term: Variance due to estimation of m(x)
        term1 = (1 / (m_x ** 2)) * (nu_2_K / (mu_2_K ** 2)) / (n * (h ** (d+2)))
        
        # Second term: Variance due to estimation of m'(x)
        term2 = ((m_prime_x ** 2) / (m_x ** 4)) * (nu_0_K / (n * h ** d))
        
        # Total asymptotic variance
        avar = (var_w / f_x) * (term1 + term2)
        
        return avar
    
    def _kernel_density(self, x):
        """Estimate the density at point x using kernel density estimation."""
        # Use statsmodels KDEMultivariate
        kde = KDEMultivariate(
            data=self.model.exog,
            var_type='c' * self.model.exog.shape[1],
            bw='normal_reference'
        )
        return kde.pdf(x)
    
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
    
    def conf_int(self, alpha=0.05, point=None):
        """Calculate confidence intervals for the estimator at a specific point."""
        if point is None:
            point = np.mean(self.model.exog, axis=0).reshape(1, -1)
            
        estimate = self.estimate_at_point(point)
        std_err = np.sqrt(self._asymptotic_variance(point))
        z_value = stats.norm.ppf(1 - alpha/2)
        
        return np.array([estimate - z_value * std_err, estimate + z_value * std_err])
    

    def estimate_at_point(self, point, variable_idx=None):
        """Calculate the estimator at a specific point."""
        # Handle variable selection (new parameter)
        if variable_idx is None:
            variable_idx = self.interest[0] if isinstance(self.interest, list) else self.interest
        
        # Get proper beta coefficient and position
        if self.all_variables:
            i = list(self.interest).index(variable_idx) if isinstance(self.interest, list) else 0
            beta_hat = self.beta_hat[variable_idx]
        else:
            i = self.interest.index(variable_idx) if isinstance(self.interest, list) else 0
            beta_hat = self.beta_hat[i] if hasattr(self.beta_hat, '__getitem__') else self.beta_hat
        
        m_at_point = self.nonparam_model.predict(point)[0]
        m_prime_at_point = self.nonparam_model.derivative(point, variable_idx)[0]  # Use variable_idx instead of self.interest
        correction = m_prime_at_point / m_at_point
        
        # Handle log_x indexing properly
        log_x_val = self.model.log_x[i] if hasattr(self.model.log_x, '__getitem__') else self.model.log_x
        
        if self.model.elasticity and log_x_val:
            return beta_hat + correction
        elif not self.model.elasticity and not log_x_val:
            return beta_hat + correction
        elif self.model.elasticity and not log_x_val:
            return beta_hat + correction * point[0, variable_idx]
        else:  # not self.model.elasticity and log_x_val
            return beta_hat + correction / point[0, variable_idx]


    def calculate_estimator_at_points_parallel(self, X_points, variable_idx=None, n_jobs=-1, backend='threading'):
        """Calculate estimator values with parallel processing (pickle-safe)."""
        import joblib
        
        if backend == 'threading':
            # Threading avoids pickling issues
            estimator_values = joblib.Parallel(n_jobs=n_jobs, backend='threading')(
                joblib.delayed(lambda x: self.estimate_at_point(x.reshape(1, -1), variable_idx))(x_point) 
                for x_point in X_points
            )
        else:
            # Multiprocessing requires module-level worker function
            args = [(x_point, self, variable_idx) for x_point in X_points]
            estimator_values = joblib.Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                joblib.delayed(_estimate_single_point_worker)(arg) for arg in args
            )
        
        return np.array(estimator_values)
    
    # Add this new method to the Results class (much cleaner - just calls estimate_at_point):
    def calculate_estimator_at_points(self, X_points, variable_idx=None):
        """Calculate estimator values at specific X points."""
        estimator_values = np.zeros(len(X_points))
        
        for j, x_point in enumerate(X_points):
            x_point = x_point.reshape(1, -1)
            estimator_values[j] = self.estimate_at_point(x_point, variable_idx)
        
        return estimator_values


    def plot_distribution(self, variable_idx=None):
        """Plot the distribution of estimator values.
        
        Parameters
        ----------
        variable_idx : int or str, optional
            Index or name of the variable to plot. If None, uses the first
            variable of interest. Defaults to None.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object for the plot
        """
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
        """Plot the correction term against the regressor of interest.
        
        Parameters
        ----------
        variable_idx : int or str, optional
            Index or name of the variable to plot. If None, uses the first
            variable of interest. Defaults to None.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object for the plot
        """
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
    n = 2000
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

