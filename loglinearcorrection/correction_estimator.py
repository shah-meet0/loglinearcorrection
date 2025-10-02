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


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def _estimate_single_point_worker(args):
    """Worker function for parallel estimation (must be at module level for pickling)."""
    x_point, results_obj, variable_idx = args
    x_point = x_point.reshape(1, -1)
    return results_obj.estimate_at_point(x_point, variable_idx)


class DoublyRobustElasticityEstimator(Model):
    r"""Doubly robust estimator for elasticities and semi-elasticities in log-transformed models.
    
    This class implements two doubly robust estimators that correct for the bias that occurs
    in log-transformed models by estimating the conditional expectation of the
    exponentiated residuals and its gradient.

    Estimator 1 (OLS-based):
    $$\hat{z_1}(x) = \hat\beta + \dfrac{\hat{m}'(x; \hat\beta)}{\hat{m}(x; \hat\beta)}$$

    Estimator 2 (PPML-based):
    $$\hat{z_2}(x) = \hat\gamma + \dfrac{\nabla \hat\psi(x)}{\hat\psi(x)}$$
    
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
    weights : array-like, optional
        Weights for weighted least squares, defaults to None

        
    Notes
    -----
    The estimators are based on the formulas:
    
    Estimator 1: \hat{z_1}(x) = \hat\beta + \frac{\hat{m}'(x; \hat\beta)}{\hat{m}(x; \hat\beta)}
    where \hat{m}(x) is estimated from OLS residuals
    
    Estimator 2: \hat{z_2}(x) = \hat\gamma + \frac{\hat\psi'(x)}{\hat\psi(x)}
    where \hat\psi(x) is estimated from PPML residuals
    """

    # TODO: Refactor names so its more consistent
    def __init__(self, endog, exog, interest=None, endog_x=None, log_x=False, instruments = None, estimator_type='kernel',
                 elasticity=False, kernel_params=None, nn_params=None, 
                 density_estimator='kernel', density_params=None, fe=None, weights=None):
        """Initialize the doubly robust estimator."""

        # Store the original data
        self.original_endog = endog
        self.original_exog = exog
        self.instruments = instruments
        self.fe = fe
        self.weights=weights
        
        # Convert pandas objects to numpy arrays and store names
        if isinstance(endog, pd.Series):
            self._endog_name = endog.name
            endog = endog.values
        else:
            self._endog_name = 'y'
            
        if isinstance(exog, pd.DataFrame):
            self._exog_names = exog.columns.tolist()
            exog = exog.values
        else:
            self._exog_names = [f'x{i}' for i in range(exog.shape[1])]

        # Process interest variable(s)
        if interest is None:
            self.interest = list(range(exog.shape[1]))
            self.all_variables = True
        else:
            self.all_variables = False
            if isinstance(interest, (list, tuple, np.ndarray)):
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

        if endog_x is None:
            if instruments is not None:
                raise ValueError('endog_x must be provided if instruments are specified')
        else:
            if isinstance(endog_x, (list, tuple, np.ndarray)):
                self.endog_x = []
                for var in endog_x:
                    if isinstance(var, str) and var in self._exog_names:
                        self.endog_x.append(self._exog_names.index(var))
                    elif isinstance(var, (int, np.integer)):
                        self.endog_x.append(var)
                    else:
                        raise ValueError(f"Endogenous variable '{var}' not recognized")
            elif isinstance(endog_x, str) and endog_x in self._exog_names:
                self.endog_x = [self._exog_names.index(endog_x)]
            else:
                self.endog_x = [endog_x]


        
        # Process fixed effects if provided
        if fe is not None:
            if isinstance(fe, list):
                if isinstance(fe[0], str):
                    self.fe_indices = self.original_exog.columns.get_indexer(fe)
                else:
                    self.fe_indices = fe
            if isinstance(fe, int):
                self.fe_indices = [fe]
            if isinstance(fe, str):
                self.fe_indices = [self.original_exog.columns.get_loc(fe)]
            self._apply_fixed_effects()
        else:
            self.endog = endog
            self.exog = exog
            self.fe_indices = None
            
        # Process log_x
        if isinstance(log_x, (list, tuple, np.ndarray)):
            if len(log_x) != len(self.interest):
                raise ValueError("log_x must be a single bool or a list matching the length of interest")
            self.log_x = log_x
        else:
            self.log_x = [log_x] * len(self.interest)

        self._handle_instruments()
            
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
        super(DoublyRobustElasticityEstimator, self).__init__(self.endog, self.exog)
    
    def _apply_fixed_effects(self):
        """Apply fixed effects transformation to the data."""
        try:
            import pyhdfe
        except ImportError:
            raise ImportError("pyhdfe is required for fixed effects estimation. "
                             "Please install pyhdfe package.")

        if self.weights is not None:
            weights = np.asarray(self.weights).reshape(-1, 1)
        else:
            weights = None

        # Extract the fixed effects columns
        if isinstance(self.original_exog, pd.DataFrame):
            fe_cols = self.original_exog.iloc[:, self.fe_indices].values
        else:
            fe_cols = self.original_exog[:, self.fe_indices]

        if fe_cols.ndim == 1:
            fe_cols = fe_cols.reshape(-1, 1)

        # Create pyhdfe algorithm
        fe_algorithm = pyhdfe.create(fe_cols, drop_singletons=False)

        # Apply demeaning to both X and y
        if isinstance(self.original_endog, pd.Series) or isinstance(self.original_endog, pd.DataFrame):
            endog_to_use = np.log(self.original_endog.copy().values.reshape(-1, 1))
        else:
            endog_to_use = np.log(self.original_endog.copy().reshape(-1, 1))

        if isinstance(self.original_exog, pd.DataFrame):
            exog_to_use = self.original_exog.copy().values
        else:
            exog_to_use = self.original_exog.copy()



        exog_to_use = np.delete(exog_to_use, self.fe_indices, axis=1).astype(np.float64)
        combined = np.column_stack([exog_to_use, endog_to_use])

        if self.instruments is not None:

            if isinstance(self.instruments, (pd.Series, pd.DataFrame)):
                Z = self.instruments.values
            else:
                Z = np.asarray(self.instruments)

            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)

            combined = np.column_stack([combined, Z])
            iv_len = Z.shape[1]
        else:
            iv_len = 0

        demeaned = fe_algorithm.residualize(combined, weights=weights)

        # Update X and y with demeaned values

        self.endog = np.exp(demeaned[:, -1 - iv_len])
        exog = demeaned[:, :-1-iv_len]
        if iv_len > 0:
            self.instruments = demeaned[:, -iv_len:]


        # Check if any exog is all 0:
        zero_cols = np.where(np.all(np.abs(exog) < 1e-6, axis=0))[0]
        self.exog = np.delete(exog, zero_cols, axis=1)

        # Check if any instruments are 0:
        if self.instruments is not None:
            zero_instrument_cols = np.where(np.all(np.abs(self.instruments) < 1e-6, axis=0))[0]
            self.instruments = np.delete(self.instruments, zero_instrument_cols, axis=1)
            if len(zero_instrument_cols) > 0:
                print('Removed zero instrument columns:', zero_instrument_cols)

        # if len(self.endog) < len(self.original_endog):
        #     print('Some singleton observations dropped. \n'
        #           'Please ensure all singleton groups removed prior to estimation if using weights.')

        # Update exog_names to remove fixed effect variables
        if hasattr(self, '_exog_names'):
            _exog_names_no_fe = []
            for i, name in enumerate(self._exog_names):
                if i not in self.fe_indices:
                    _exog_names_no_fe.append(name)
                else:
                    print(f"Removing fixed effect variable: {name}")

            _exog_names = []
            for i, name in enumerate(_exog_names_no_fe):
                if i not in zero_cols:
                    _exog_names.append(name)
                else:
                    print(f"Removing redundant column variable: {name}")

            self._exog_names = np.array(_exog_names)

        # Update interest index if needed
        if hasattr(self, 'interest'):
            if len(self.interest) == 1:
                adjusted_index = self.interest[0]
                for idx in sorted(self.fe_indices):
                    if idx < self.interest[0]:
                        adjusted_index -= 1

                readjusted_index = adjusted_index
                for idx in sorted(zero_cols):
                    if idx < readjusted_index:
                        adjusted_index -= 1

                if adjusted_index != self.interest[0]:
                    print('Adjusting interest index from', self.interest[0], 'to', adjusted_index)
                    self.interest = [adjusted_index]
            else:
                raise NotImplementedError('Fixed effects not implemented for multiple variables of interest')

        # Update endog_x if needed
        if hasattr(self, 'endog_x'):
            adjusted_endog_x = []

            for idx in sorted(self.endog_x):
                adjusted_index= idx
                for fe_idx in sorted(self.fe_indices):
                    if fe_idx < idx:
                        adjusted_index -= 1

                if adjusted_index in zero_cols:
                    print(f'Endogenous variable {idx} dropped.')
                    continue
                else:
                    for zero_idx in sorted(zero_cols):
                        if zero_idx < adjusted_index:
                            adjusted_index -= 1
                    adjusted_endog_x.append(adjusted_index)

            self.endog_x = adjusted_endog_x


        print('Fixed effects applied. Variables have been demeaned.')

    def _handle_instruments(self):
        """Instrument the endogenous variable using instruments and controls (all demeaned)."""
        if self.instruments is None:
            return  # no IV logic needed

        if self.weights is None:
            weights = 1.
        else:
            weights = np.asarray(self.weights).ravel()

        idx_endog = self.endog_x

        # Extract components
        controls_idx = [i for i in range(self.exog.shape[1]) if i not in idx_endog]

        # Build X for first-stage: instruments + controls (all demeaned)
        X_first_stage = np.column_stack([self.instruments, self.exog[:, controls_idx]])

        # For each endogenous regressor, run first stage OLS
        for i in idx_endog:
            y_first_stage = self.exog[:, i]
            reg = sm.WLS(y_first_stage, X_first_stage, weights).fit()
            fitted_vals = reg.fittedvalues
            # Replace the endogenous regressor with instrumented version
            self.exog[:, i] = fitted_vals

        # Replace the endogenous regressor with instrumented version

        print(f"Instrumented variable at columns {idx_endog}")


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
        
    def fit(self, method='ols', bootstrap=False, bootstrap_reps=500,
        bootstrap_method='pairs', compute_asymptotic_variance=None, **kwargs):
        """Fit the doubly robust estimator.
        
        Parameters
        ----------
        method : str, optional
            Method to use for the parametric component: 'ols' (Estimator 1) or 'ppml' (Estimator 2)
        bootstrap : bool, optional
            Whether to compute bootstrap standard errors, defaults to False
        bootstrap_reps : int, optional
            Number of bootstrap replications if bootstrap=True, defaults to 500
        bootstrap_method : str, optional
            Bootstrap method to use, either 'pairs' (resample observation pairs) or 
            'residuals' (resample residuals), defaults to 'pairs'
        compute_asymptotic_variance : bool, optional
            Whether to compute asymptotic variance. If None, defaults to True when 
            bootstrap=False and False when bootstrap=True
        **kwargs : dict
            Additional arguments to pass to the parametric fit method
            
        Returns
        -------
        DoublyRobustElasticityEstimatorResults
            Fitted model results object
        """

        weights = self.weights

        # Set default for asymptotic variance computation
        if compute_asymptotic_variance is None:
            compute_asymptotic_variance = not bootstrap

        # Step 1: Fit the parametric model
        if method == 'ols':
            kwargs.setdefault("cov_type", 'HC3')
            parametric_results = self._fit_base_ols(weights, **kwargs)
            # Calculate OLS residuals: u_i = log(y_i) - x_i * beta_hat
            residuals = parametric_results.resid
            
        elif method == 'ppml':
            kwargs.setdefault("cov_type", 'HC3')
            parametric_results = self._fit_base_ppml(weights, **kwargs)
            # Calculate PPML residuals: u_i = log(y_i) - x_i * gamma_hat
            # fitted_values from GLM are exp(x_i * gamma_hat)
            residuals = np.log(self.endog) - np.log(parametric_results.fittedvalues)

        else:
            raise ValueError(f"Method '{method}' not supported. Use 'ols' or 'ppml'.")
            
        # Print parametric results for variables of interest
        if self.all_variables:
            print(f'{method.upper()} model fitted for all variables')
        elif len(self.interest) == 1:
            var_idx = self.interest[0]
            var_name = self.exog_names[var_idx]
            coef_name = 'beta' if method == 'ols' else 'gamma'
            print(f'{method.upper()} model fitted, estimated {coef_name} for {var_name}: {parametric_results.params[var_idx]:.4f}')
        else:
            print(f'{method.upper()} model fitted for {len(self.interest)} variables of interest')
            for var_idx in self.interest:
                var_name = self.exog_names[var_idx]
                coef_name = 'beta' if method == 'ols' else 'gamma'
                print(f'  {var_name}: {parametric_results.params[var_idx]:.4f}')
            
        # Step 2: Calculate exponentiated residuals
        exp_residuals = np.exp(residuals)
        
        print('Estimating correction term...')
        
        # Step 3: Fit the nonparametric model for E[exp(u)|X]
        if self.estimator_type == 'kernel':
            nonparam_model = self._fit_kernel_model(exp_residuals)
        elif self.estimator_type == 'nn':
            nonparam_model = self._fit_nn_model(exp_residuals)
        elif self.estimator_type == 'ols':
            nonparam_model = self._fit_ols_model(exp_residuals)
        elif self.estimator_type == 'binary':
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
        
        # Step 4: Fit the density estimator if needed for asymptotic variance
        density_model = None
        if compute_asymptotic_variance and self.density_estimator == 'nn':
            print('Fitting neural network density estimator...')
            density_model = self._fit_nn_density_estimator(self.exog)
            print('Neural network density estimator fitted')
            
        print('Correction term estimated')
        
        # Step 5: Create the results object
        results = DoublyRobustElasticityEstimatorResults(
            model=self,
            parametric_results=parametric_results,
            parametric_method=method,
            nonparam_model=nonparam_model,
            exp_residuals=exp_residuals,
            density_model=density_model,
            compute_asymptotic_variance=compute_asymptotic_variance
        )
        
        # Step 6: Compute bootstrap standard errors if requested
        if bootstrap:
            print(f'Computing bootstrap standard errors with {bootstrap_reps} replications...')
            results._bootstrap_standard_errors(bootstrap_reps, bootstrap_method, weights, kwargs, method)
            print('Bootstrap completed')
            
        return results

    def _fit_base_ols(self, weights, **kwargs):
        if weights is None:
            return sm.OLS(np.log(self.endog), self.exog).fit(**kwargs)
        else:
            return sm.WLS(np.log(self.endog), self.exog, weights=weights).fit(**kwargs)

    def _fit_base_ppml(self, weights, **kwargs):
        if weights is None:
            parametric_results = sm.GLM(self.endog, self.exog,
                                        family=sm.families.Poisson()).fit(**kwargs)
        else:
            parametric_results = sm.GLM(self.endog, self.exog,
                                        family=sm.families.Poisson(),
                                        freq_weights=weights).fit(**kwargs)
        return parametric_results

    def _fit_kernel_model(self, exp_residuals):
        """Fit a local linear kernel regression model for E[exp(u)|X]."""
        var_type = self.kernel_params.get('var_type', 'c' * self.exog.shape[1])
        bw = self.kernel_params.get('bw', 'cv_ls')
        
        # Normalize exponentiated residuals for numerical stability
        min_exp_r = np.min(exp_residuals)
        normalized_exp_residuals = exp_residuals / min_exp_r
        
        kr_model = KernelReg(
            endog=normalized_exp_residuals,
            exog=self.exog,
            var_type=var_type,
            reg_type='ll',
            bw=bw
        )
        
        return KernelRegressionModel(kr_model, self.binary_vars, self.ordinal_vars, min_exp_r)
    
    def _fit_nn_model(self, exp_residuals):
        """Fit a neural network model for E[exp(u)|X]."""
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except ImportError:
            raise ImportError("TensorFlow is required for neural network estimator.")
        
        def setup_dre_gpu():


            """Force DRE to use GPU."""

            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')

                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ DRE GPU setup: {len(gpus)} GPU(s) available")
                    return True
                return False
            except:
                return False

        dre_gpu_available = setup_dre_gpu()

        if dre_gpu_available:
            device_name = '/GPU:0'
        else:
            device_name = '/CPU:0'


        
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
        
        nn_params = {**defaults, **self.nn_params}

        # CRITICAL: Everything must be created within the same device context
        with tf.device(device_name):
            # Prepare data on the target device
            min_exp_r = np.min(exp_residuals)
            normalized_exp_residuals = exp_residuals / min_exp_r
            
            residuals_mean = np.mean(normalized_exp_residuals)
            residuals_std = np.std(normalized_exp_residuals)
            scaled_residuals = (normalized_exp_residuals - residuals_mean) / residuals_std
            
            X_nn = self.exog.copy()
            
            # Create tensors on target device
            X_tensor = tf.convert_to_tensor(X_nn, dtype=tf.float32)
            y_tensor = tf.convert_to_tensor(scaled_residuals, dtype=tf.float32)
            
            # Create model within device context
            model = self._create_nn_model(nn_params, device_name)
            
            # Initialize weights on correct device
            dummy_input = tf.zeros((1, self.exog.shape[1]), dtype=tf.float32)
            _ = model(dummy_input)  # Force weight initialization

            # Fixed device info printing - no direct .device access
            try:
                if model.weights:
                    print(f"Model initialized with {len(model.weights)} weight tensors")
                else:
                    print("No model weights found")
                print(f"Training tensors ready on {device_name}")
            except Exception as e:
                print(f"Model info: weights initialized, training on {device_name}") 
            # Training within device context
            history = model.fit(
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
        
        print(f"Training completed on {device_name}")
        
        # Return patched model
        return NNRegressionModel(model, residuals_mean, residuals_std, min_exp_r, 
                        self.binary_vars, self.ordinal_vars)
    
    def _fit_binary_model(self, exp_residuals):
        """Fit a binary model for E[exp(u)|X] when the regressor of interest is binary."""
        min_exp_r = np.min(exp_residuals)
        normalized_exp_residuals = exp_residuals / min_exp_r
        
        binary_interest = [var_idx for var_idx in self.interest if var_idx in self.binary_vars]
        
        if not binary_interest and self.interest:
            binary_interest = [self.interest[0]]
        
        binary_models = {}
        for var_idx in binary_interest:
            x_binary = self.exog[:, var_idx]
            eu_0 = np.mean(normalized_exp_residuals[x_binary == 0])
            eu_1 = np.mean(normalized_exp_residuals[x_binary == 1])
            binary_models[var_idx] = (eu_0, eu_1)
        
        return MultiBinaryRegressionModel(binary_models, min_exp_r, 
                                         self.binary_vars, self.ordinal_vars)

    def _fit_ols_model(self, exp_residuals):
        """Fit a polynomial OLS model for E[exp(u)|X]."""
        degree = self.kernel_params.get('degree', 3)
        
        min_exp_r = np.min(exp_residuals)
        normalized_exp_residuals = exp_residuals / min_exp_r
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(self.exog)
        
        ols_model = sm.OLS(normalized_exp_residuals, X_poly).fit()
        
        print(f'OLS correction model fitted with polynomial degree {degree}')
        print(f'R-squared: {ols_model.rsquared:.4f}')
        
        return OLSRegressionModel(ols_model, poly, min_exp_r, 
                                 self.binary_vars, self.ordinal_vars)
    
    def _create_nn_model(self, nn_params,device_name):
        """Create a neural network model based on the provided parameters."""
        import tensorflow as tf
        
        num_layers = nn_params['num_layers']
        activation = nn_params['activation']
        num_units = nn_params['num_units']
        optimizer = nn_params['optimizer']
        loss = nn_params['loss']
        
        if isinstance(num_units, int):
            num_units = [num_units] * num_layers
        elif len(num_units) != num_layers:
            raise ValueError("num_units must be an int or a list of length num_layers")
        
        # Create model within device context
        with tf.device(device_name):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=(self.exog.shape[1],)))
            
            for i in range(num_layers):
                model.add(tf.keras.layers.Dense(
                    num_units[i], 
                    activation=activation,
                    kernel_initializer='glorot_uniform'
                ))
            
            model.add(tf.keras.layers.Dense(1))
            model.compile(optimizer=optimizer, loss=loss)
        
        return model

    def _fit_nn_density_estimator(self, X):
        """Fit neural network density estimator for f_X(x)."""
        return NNDensityEstimator(X, **self.density_params)


class DoublyRobustElasticityEstimatorResults(Results):
    """
    Results from the DoublyRobustElasticityEstimator supporting both OLS and PPML methods.
    """
    
    def __init__(self, model, parametric_results, parametric_method, nonparam_model, exp_residuals, 
                 density_model=None, compute_asymptotic_variance=True, 
                 bootstrap=False, bootstrap_reps=None, bootstrap_estimates=None, 
                 kernel_type='gaussian', density_bandwidth_method='scott'):
        """Initialize the results object."""
        self.model = model
        self.parametric_results = parametric_results
        self.parametric_method = parametric_method.lower()
        self.nonparam_model = nonparam_model
        self.exp_residuals = exp_residuals
        self.interest = model.interest
        self.all_variables = model.all_variables
        self.density_model = density_model
        self.compute_asymptotic_variance = compute_asymptotic_variance
        
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
        
        # Extract the parametric estimates (beta for OLS, gamma for PPML)
        if self.all_variables:
            self.parametric_coef = parametric_results.params
        else:
            self.parametric_coef = [parametric_results.params[idx] for idx in self.interest]
        
        # For backward compatibility, also store as beta_hat
        self.beta_hat = self.parametric_coef
        
        # Initialize asymptotic variance components only if requested
        if self.compute_asymptotic_variance:
            self._initialize_asymptotic_variance()
        else:
            self.kernel_constants = None
            self._conditional_variance_model = None
        
        # Compute correction terms for all data points
        self._compute_corrections()
        
        # Initialize the base Results class
        super(DoublyRobustElasticityEstimatorResults, self).__init__(model, parametric_results.params)
    
    def _initialize_asymptotic_variance(self):
        """Initialize components needed for asymptotic variance calculation."""
        self.kernel_constants = self._compute_kernel_constants()
        self._conditional_variance_model = None
        self._fit_conditional_variance_model()
    
    def _compute_kernel_constants(self):
        """Compute kernel constants for multivariate asymptotic variance calculation."""
        d = self.n_vars
        
        if self.kernel_type == 'gaussian':
            nu_0_K0 = 1 / (2 * np.sqrt(np.pi))
            mu_2_K0 = 1.0
            nu_2_K0 = 3 / (4 * np.sqrt(np.pi))
        elif self.kernel_type == 'epanechnikov':
            nu_0_K0 = 3/5
            mu_2_K0 = 1/5  
            nu_2_K0 = 3/35
        else:
            warnings.warn(f"Unknown kernel type '{self.kernel_type}'. Using Gaussian defaults.")
            nu_0_K0 = 1 / (2 * np.sqrt(np.pi))
            mu_2_K0 = 1.0
            nu_2_K0 = 3 / (4 * np.sqrt(np.pi))
        
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
        """Fit nonparametric model for conditional variance σ²_w(x) = Var(e^u|X=x)."""
        if self.model.estimator_type == 'kernel':
            var_type = self.model.kernel_params.get('var_type', 'c' * self.n_vars)
            bw = self.model.kernel_params.get('bw', 'cv_ls')
            
            exp_residuals_squared = self.exp_residuals ** 2
            
            self._conditional_variance_model = KernelReg(
                endog=exp_residuals_squared,
                exog=self.model.exog,
                var_type=var_type,
                reg_type='ll',
                bw=bw
            )
        elif self.model.estimator_type == 'ols':
            degree = self.model.kernel_params.get('degree', 3)
            self._poly_variance = PolynomialFeatures(degree=degree, include_bias=False)
            
            exp_residuals_squared = self.exp_residuals ** 2
            X_poly = self._poly_variance.fit_transform(self.model.exog)
            self._conditional_variance_model = sm.OLS(exp_residuals_squared, X_poly).fit()
        elif self.model.estimator_type == 'nn':
            print("Fitting neural network for conditional variance estimation...")
            self._conditional_variance_model = self._fit_nn_conditional_variance_model()
            print("Neural network conditional variance model fitted successfully!")
        else:
            exp_residuals_squared = self.exp_residuals ** 2
            var_type = 'c' * self.n_vars
            
            self._conditional_variance_model = KernelReg(
                endog=exp_residuals_squared,
                exog=self.model.exog,
                var_type=var_type,
                reg_type='ll',
                bw='cv_ls'
            )

    def _fit_nn_conditional_variance_model(self):
        """Fit neural network model for conditional variance estimation."""
        exp_residuals_squared = self.exp_residuals ** 2
        
        density_params = dict(self.model.density_params)
        
        if 'hidden_layers' in density_params:
            density_params['hidden_layers'] = [max(16, int(0.75 * units)) 
                                              for units in density_params['hidden_layers']]
        
        density_params['target_type'] = 'conditional_variance'
        density_params.setdefault('verbose', 0)
        
        try:
            nn_estimator = NNDensityEstimator(
                X=self.model.exog,
                y=exp_residuals_squared,
                **density_params
            )
            return nn_estimator
        except Exception as e:
            warnings.warn(f"Neural network conditional variance training failed: {e}. "
                         f"Falling back to kernel method.")
            return self._fit_kernel_conditional_variance_fallback()

    def _fit_kernel_conditional_variance_fallback(self):
        """Fallback method using kernel regression when neural network fails."""
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
        """Estimate conditional variance σ²_w(x) = Var(e^u|X=x) at point x."""
        if not self.compute_asymptotic_variance:
            raise ValueError("Asymptotic variance computation is disabled.")
        
        x = np.atleast_2d(x)
        if x.shape[0] != 1:
            raise ValueError("x must be a single point of shape (1, d)")
        
        # Get E[w²|X=x]
        if self.model.estimator_type == 'ols' and hasattr(self, '_poly_variance'):
            x_poly = self._poly_variance.transform(x)
            m_w2_x = self._conditional_variance_model.predict(x_poly)[0]
        elif hasattr(self._conditional_variance_model, 'predict_density'):
            m_w2_x = self._conditional_variance_model.predict_density(x)[0]
        else:
            m_w2_x = self._conditional_variance_model.fit(x)[0][0]
        
        # Get E[w|X=x] = m(x)
        m_x = self.nonparam_model.predict(x)[0]
        
        # Compute conditional variance with non-negativity constraint
        conditional_var = max(0.0, m_w2_x - m_x**2)
        
        return conditional_var

    def _estimate_density(self, x):
        """Estimate density f_X(x) at point x."""
        if not self.compute_asymptotic_variance:
            raise ValueError("Asymptotic variance computation is disabled.")
        
        x = np.atleast_2d(x)
        
        if self.density_model is not None:
            return self.density_model.predict_density(x)[0]
        else:
            var_type = 'c' * self.n_vars
            kde = KDEMultivariate(
                data=self.model.exog,
                var_type=var_type,
                bw=self.density_bandwidth_method
            )
            return kde.pdf(x.flatten())
    
    def _get_bandwidth(self):
        """Get bandwidth used in the nonparametric model."""
        if hasattr(self.nonparam_model, 'bandwidth') and self.nonparam_model.bandwidth is not None:
            bandwidth = self.nonparam_model.bandwidth
            if hasattr(bandwidth, '__len__') and len(bandwidth) > 1:
                return np.mean(bandwidth)
            else:
                return float(bandwidth)
        else:
            return self.n_obs ** (-1 / (4 + self.n_vars))
    
    def _compute_corrections(self):
        """Compute correction terms for all data points in the sample."""
        # Get the values of m(x) or psi(x) for all observations
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
                # For binary variables, use the formula: e^{coef} * (m1/m0) - 1
                if self.all_variables:
                    param_coef = self.parametric_coef[var_idx]
                else:
                    param_coef = self.parametric_coef[i]
                
                # Get m0 and m1 values
                X_0 = self.model.exog.copy()
                X_1 = self.model.exog.copy()
                X_0[:, var_idx] = 0
                X_1[:, var_idx] = 1
                m_0 = self.nonparam_model.predict(X_0)
                m_1 = self.nonparam_model.predict(X_1)
                
                # For binary variables, the semi-elasticity is e^{coef} * (m1/m0) - 1
                semi_elasticity = np.exp(param_coef) * (m_1 / m_0) - 1
                self.estimator_values[var_idx] = semi_elasticity
                
                # Store correction term for consistency
                self.correction[var_idx] = self.m_prime_hat[var_idx] / self.m_hat
            else:
                # For continuous variables, use the appropriate formula
                # Compute the correction term
                self.correction[var_idx] = self.m_prime_hat[var_idx] / self.m_hat
                
                # Compute the doubly robust estimator values
                if self.model.elasticity and self.model.log_x[i]:
                    # For elasticity with log(x)
                    self.estimator_values[var_idx] = self.parametric_coef[i] + self.correction[var_idx]
                elif not self.model.elasticity and not self.model.log_x[i]:
                    # For semi-elasticity with untransformed x
                    self.estimator_values[var_idx] = self.parametric_coef[i] + self.correction[var_idx]
                elif self.model.elasticity and not self.model.log_x[i]:
                    # For elasticity with untransformed x
                    self.estimator_values[var_idx] = self.parametric_coef[i] + self.correction[var_idx] * self.model.exog[:, var_idx]
                else:  # not self.model.elasticity and self.model.log_x[i]
                    # For semi-elasticity with log(x)
                    x_val_col = self.model.exog[:, var_idx].copy()
                    x_val_col[x_val_col == 0] = 1e-9
                    self.estimator_values[var_idx] = self.parametric_coef[i] + self.correction[var_idx] / x_val_col

    def asymptotic_variance(self, x, variable_idx=None):
        """Compute scalar asymptotic variance for specific variable at point x."""
        if not self.compute_asymptotic_variance:
            raise ValueError("Asymptotic variance computation is disabled.")
        
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
        term1 = (m_x**2 * K_1_scalar) / (self.n_obs * h**(self.n_vars + 2))
        term2 = (h**2 * m_prime_x**2 * K_0_scalar) / (self.n_obs * h**self.n_vars)
        
        # Common factor
        common_factor = sigma_w2_x / (f_x * m_x**4)
        
        avar_scalar = common_factor * (term1 + term2)
        
        return avar_scalar
    
    def asymptotic_standard_error_at_point(self, x, variable_idx=None):
        """Compute asymptotic standard error at point x."""
        variance = self.asymptotic_variance(x, variable_idx)
        return np.sqrt(variance)
    
    def asymptotic_standard_error_at_average(self, variable_idx=None):
        """Compute asymptotic standard error at average values of regressors."""
        X_mean = np.mean(self.model.exog, axis=0).reshape(1, -1)
        return self.asymptotic_standard_error_at_point(X_mean, variable_idx)
    
    def asymptotic_confidence_interval(self, x, variable_idx=None, alpha=0.05):
        """Compute asymptotic confidence interval at point x."""
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

    def _bootstrap_standard_errors(self, bootstrap_reps, bootstrap_method, weights, kwargs, method):
        """Compute bootstrap standard errors.
        
        Modified version that stores full DREER objects to enable standard error
        calculation at arbitrary points.
        """
        if bootstrap_method != 'pairs':
            raise ValueError("Only 'pairs' bootstrap is available.")
        
        n = len(self.model.endog)
        
        if self.all_variables:
            var_indices = self.interest
        else:
            var_indices = self.interest
            
        # Store the full DREER objects from bootstrap
        self.bootstrap_results = []
        
        # Also keep the summary statistics for backward compatibility
        bootstrap_estimates_dict = {var_idx: np.zeros((bootstrap_reps, 4)) 
                                   for var_idx in var_indices}
        
        for i in range(bootstrap_reps):
            if i > 0 and i % 100 == 0:
                print(f"  Completed {i} bootstrap replications")
                
            # Resample pairs (x_i, y_i)
            indices = np.random.choice(n, size=n, replace=True)
            bs_exog = self.model.exog[indices]
            bs_endog = self.model.endog[indices]

            clean_nn_params = self.model.nn_params.copy()
            if 'model' in clean_nn_params:
                del clean_nn_params['model']

            
            # Create a new estimator with the resampled data
            bs_model = DoublyRobustElasticityEstimator(
                endog=bs_endog,
                exog=bs_exog,
                interest=self.model.interest,
                log_x=self.model.log_x,
                estimator_type=self.model.estimator_type,
                elasticity=self.model.elasticity,
                kernel_params=self.model.kernel_params,
                nn_params=clean_nn_params,
                density_estimator=self.model.density_estimator,
                density_params=self.model.density_params,
                weights = weights
            )
            
            # Fit the bootstrapped model with the specified method
            bs_results = bs_model.fit(method=method, bootstrap=False, 
                                    compute_asymptotic_variance=False, **kwargs)

            # 💡 Create and store the lean result object
            lean_result = LeanBootstrapResult(
                parametric_params=bs_results.parametric_results.params,
                nonparam_model=bs_results.nonparam_model,
                model_interest_indices=bs_model.interest,
                model_log_x_flags=bs_model.log_x,
                model_binary_vars=bs_model.binary_vars
            )
            self.bootstrap_results.append(lean_result)

            
            # Also store summary estimates for backward compatibility
            for var_idx in var_indices:
                if bs_results.all_variables:
                    var_pos = list(bs_results.interest).index(var_idx)
                    param_coef = bs_results.parametric_coef[var_idx]
                else:
                    var_pos = bs_results.interest.index(var_idx)
                    param_coef = bs_results.parametric_coef[var_pos]
                
                original_X_mean = np.mean(self.model.exog, axis=0).reshape(1, -1)

                bootstrap_estimates_dict[var_idx][i, 0] = param_coef
                bootstrap_estimates_dict[var_idx][i, 1] = np.mean(bs_results.correction[var_idx])
                
                original_estimator_values = bs_results.calculate_estimator_at_points(
                    self.model.exog, var_idx 
                )
                bootstrap_estimates_dict[var_idx][i, 2] = np.mean(original_estimator_values)
                bootstrap_estimates_dict[var_idx][i, 3] = bs_results.estimate_at_point(
                    original_X_mean, var_idx
                )
        
        self.bootstrap = True
        self.bootstrap_reps = bootstrap_reps
        self.bootstrap_estimates_dict = bootstrap_estimates_dict
        
        # Calculate bootstrap standard errors for each variable (backward compatibility)
        self.bootstrap_se_dict = {var_idx: np.std(bootstrap_estimates_dict[var_idx], axis=0) 
                                 for var_idx in var_indices}


    def bootstrap_standard_error_at_point(self, X_point, variable_idx=None):
        """Calculate bootstrap standard error at a specific point.
        
        This new method uses the stored DREER objects to compute standard errors
        at arbitrary points.
        
        Parameters
        ----------
        X_point : array-like of shape (1, d) or (d,)
            Point at which to compute the standard error
        variable_idx : int or str, optional
            Variable for which to compute standard error.
            If None, uses first variable of interest.
        
        Returns
        -------
        float
            Bootstrap standard error at the specified point
            
        Raises
        ------
        ValueError
            If bootstrap has not been performed or bootstrap_results not stored
        """
        if not self.bootstrap:
            raise ValueError("Bootstrap standard errors have not been computed. "
                            "Call fit() with bootstrap=True first.")
        
        if not hasattr(self, 'bootstrap_results'):
            raise ValueError("Bootstrap results not stored. This may be an older "
                            "bootstrap run. Re-run bootstrap to store full results.")
        
        # Handle variable selection
        if variable_idx is None:
            variable_idx = self.interest[0] if isinstance(self.interest, list) else self.interest
        elif isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        
        # Ensure X_point is properly shaped
        X_point = np.atleast_2d(X_point)
        if X_point.shape[0] != 1:
            X_point = X_point.reshape(1, -1)
        
        # Get estimates at this point from all bootstrap samples
        bootstrap_estimates_at_point = []
        for bs_result in self.bootstrap_results:
            try:
                # Reconstruct the elasticity estimate using the lean object
                current_param_coef = bs_result.parametric_params[variable_idx]
                
                try:
                    # Find if the variable of interest was log-transformed in the bootstrap model
                    idx_in_model_interest_list = bs_result.model_interest_indices.index(variable_idx)
                    current_log_x_bool = bs_result.model_log_x_flags[idx_in_model_interest_list]
                except ValueError:
                    current_log_x_bool = False

                if variable_idx in bs_result.model_binary_vars:
                    X_0 = X_point.copy(); X_1 = X_point.copy()
                    X_0[:, variable_idx] = 0; X_1[:, variable_idx] = 1
                    m_0 = bs_result.nonparam_model.predict(X_0)
                    m_1 = bs_result.nonparam_model.predict(X_1)
                    m_0[m_0 == 0] = 1e-9
                    estimate = np.exp(current_param_coef) * (m_1 / m_0) - 1
                else:
                    m_at_point = bs_result.nonparam_model.predict(X_point)
                    m_prime_at_point = bs_result.nonparam_model.derivative(X_point, variable_idx)
                    m_at_point[m_at_point == 0] = 1e-9
                    correction = m_prime_at_point / m_at_point
                    
                    # This logic must exactly match estimate_at_point
                    if self.model.elasticity and current_log_x_bool:
                        estimate = current_param_coef + correction
                    elif not self.model.elasticity and not current_log_x_bool:
                        estimate = current_param_coef + correction
                    elif self.model.elasticity and not current_log_x_bool:
                        estimate = current_param_coef + correction * X_point[:, variable_idx]
                    else:
                        x_val_col = X_point[:, variable_idx].copy()
                        x_val_col[x_val_col == 0] = 1e-9
                        estimate = current_param_coef + correction / x_val_col
                
                bootstrap_estimates_at_point.append(estimate[0])
            except Exception as e:
                # Handle cases where bootstrap sample might fail at this point
                # (e.g., numerical issues)
                print(f"Warning: Bootstrap sample failed at point: {e}")
                continue
        
        if len(bootstrap_estimates_at_point) == 0:
            raise ValueError("All bootstrap samples failed at the specified point")
        
        # Return standard error
        return np.std(bootstrap_estimates_at_point)


    def bootstrap_confidence_interval_at_point(self, X_point, variable_idx=None, 
                                              alpha=0.05, method='percentile'):
        """Calculate bootstrap confidence interval at a specific point.
        
        This new method uses the stored DREER objects to compute confidence
        intervals at arbitrary points.
        
        Parameters
        ----------
        X_point : array-like of shape (1, d) or (d,)
            Point at which to compute the confidence interval
        variable_idx : int or str, optional
            Variable for which to compute confidence interval.
            If None, uses first variable of interest.
        alpha : float, default 0.05
            Significance level for (1-α)×100% confidence interval
        method : str, default 'percentile'
            Method for calculating confidence intervals: 'percentile' or 'normal'
        
        Returns
        -------
        tuple of float
            (lower_bound, upper_bound) for the confidence interval
            
        Raises
        ------
        ValueError
            If bootstrap has not been performed or bootstrap_results not stored
        """
        if not self.bootstrap:
            raise ValueError("Bootstrap standard errors have not been computed. "
                            "Call fit() with bootstrap=True first.")
        
        if not hasattr(self, 'bootstrap_results'):
            raise ValueError("Bootstrap results not stored. This may be an older "
                            "bootstrap run. Re-run bootstrap to store full results.")
        
        # Handle variable selection
        if variable_idx is None:
            variable_idx = self.interest[0] if isinstance(self.interest, list) else self.interest
        elif isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        
        # Ensure X_point is properly shaped
        X_point = np.atleast_2d(X_point)
        if X_point.shape[0] != 1:
            X_point = X_point.reshape(1, -1)
        
        # Get estimates at this point from all bootstrap samples
        bootstrap_estimates_at_point = []
        for bs_result in self.bootstrap_results:
            try:
                estimate = bs_result.estimate_at_point(X_point, variable_idx)
                bootstrap_estimates_at_point.append(estimate)
            except Exception:
                continue
        
        if len(bootstrap_estimates_at_point) == 0:
            raise ValueError("All bootstrap samples failed at the specified point")
        
        bootstrap_estimates_at_point = np.array(bootstrap_estimates_at_point)
        
        if method == 'percentile':
            # Percentile method
            lower = np.percentile(bootstrap_estimates_at_point, 100 * alpha / 2)
            upper = np.percentile(bootstrap_estimates_at_point, 100 * (1 - alpha / 2))
            
        elif method == 'normal':
            # Normal approximation method
            point_estimate = self.estimate_at_point(X_point, variable_idx)
            std_error = np.std(bootstrap_estimates_at_point)
            z_value = stats.norm.ppf(1 - alpha/2)
            
            lower = point_estimate - z_value * std_error
            upper = point_estimate + z_value * std_error
            
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'percentile' or 'normal'.")
        
        return (lower, upper)


    def get_bootstrap_distribution_at_point(self, X_point, variable_idx=None):
        """Get the full bootstrap distribution of estimates at a specific point.
        
        Parameters
        ----------
        X_point : array-like of shape (1, d) or (d,)
            Point at which to get the bootstrap distribution
        variable_idx : int or str, optional
            Variable for which to get distribution.
            If None, uses first variable of interest.
        
        Returns
        -------
        np.ndarray
            Array of bootstrap estimates at the specified point
            
        Raises
        ------
        ValueError
            If bootstrap has not been performed or bootstrap_results not stored
        """
        if not self.bootstrap:
            raise ValueError("Bootstrap standard errors have not been computed. "
                            "Call fit() with bootstrap=True first.")
        
        if not hasattr(self, 'bootstrap_results'):
            raise ValueError("Bootstrap results not stored. This may be an older "
                            "bootstrap run. Re-run bootstrap to store full results.")
        
        # Handle variable selection
        if variable_idx is None:
            variable_idx = self.interest[0] if isinstance(self.interest, list) else self.interest
        elif isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        
        # Ensure X_point is properly shaped
        X_point = np.atleast_2d(X_point)
        if X_point.shape[0] != 1:
            X_point = X_point.reshape(1, -1)
        
        # Get estimates at this point from all bootstrap samples
        bootstrap_estimates_at_point = []
        for bs_result in self.bootstrap_results:
            try:
                estimate = bs_result.estimate_at_point(X_point, variable_idx)
                bootstrap_estimates_at_point.append(estimate)
            except Exception:
                continue
        
        return np.array(bootstrap_estimates_at_point)

    def summary(self):
        """Provide a summary of the results."""
        from statsmodels.iolib.summary2 import Summary
        
        smry = Summary()
        
        # Add a title
        method_name = "OLS-based" if self.parametric_method == 'ols' else "PPML-based"
        if self.model.elasticity:
            title = f"Doubly Robust Elasticity Estimator Results ({method_name})"
        else:
            title = f"Doubly Robust Semi-Elasticity Estimator Results ({method_name})"
            
        smry.add_title(title)
        
        # Add parametric results
        method_upper = self.parametric_method.upper()
        smry.add_text(f"{method_upper} Results:")
        
        # Get the parametric summary table
        param_summary_table = self.parametric_results.summary().tables[1]
        if hasattr(param_summary_table, 'data'):
            param_data = []
            for row in param_summary_table.data[1:]:
                param_data.append(row)
            param_df = pd.DataFrame(param_data, columns=param_summary_table.data[0])
            smry.add_df(param_df)
        else:
            smry.add_text(str(param_summary_table))
        
        # Build a table of corrected estimates for each variable of interest
        data = []
        
        for i, var_idx in enumerate(self.interest):
            var_name = self.model.exog_names[var_idx]
            row = {}
            row["Variable"] = var_name
            
            # Add parametric estimate
            if self.all_variables:
                param_coef = self.parametric_coef[var_idx]
            else:
                param_coef = self.parametric_coef[i]
            
            coef_name = "Beta" if self.parametric_method == 'ols' else "Gamma"
            row[f"{coef_name} Estimate"] = f"{param_coef:.4f}"
            
            # Add average correction
            avg_corr = np.mean(self.correction[var_idx])
            row["Average Correction"] = f"{avg_corr:.4f}"
            
            # Add average estimate
            avg_est = np.mean(self.estimator_values[var_idx])
            row["Average Estimate"] = f"{avg_est:.4f}"
            
            # Add estimate at average
            est_at_avg = self.estimate_at_average(var_idx)
            row["Estimate at Average"] = f"{est_at_avg:.4f}"
            
            # Add standard errors based on method used
            if self.bootstrap:
                if hasattr(self, 'bootstrap_se_dict'):
                    se = self.bootstrap_se_dict[var_idx][3]
                    row["Bootstrap SE"] = f"{se:.4f}"
                    row["95% CI Lower"] = f"{est_at_avg - 1.96 * se:.4f}"
                    row["95% CI Upper"] = f"{est_at_avg + 1.96 * se:.4f}"
                else:
                    row["Bootstrap SE"] = "N/A"
                    row["95% CI Lower"] = "N/A"
                    row["95% CI Upper"] = "N/A"
            elif self.compute_asymptotic_variance:
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
            else:
                row["SE"] = "Not computed"
                row["95% CI Lower"] = "N/A"
                row["95% CI Upper"] = "N/A"
            
            data.append(row)
        
        results_df = pd.DataFrame(data)
        results_df = results_df.set_index("Variable")
        
        smry.add_text(f"\nDoubly Robust Estimates (Estimator {1 if self.parametric_method == 'ols' else 2}):")
        smry.add_df(results_df)
        
        # Add information about the nonparametric component
        smry.add_text(f"\nNonparametric Component: {self.model.estimator_type}")
        if self.model.estimator_type == 'kernel' and hasattr(self.nonparam_model, 'bandwidth'):
            smry.add_text(f"Bandwidth: {self.nonparam_model.bandwidth}")
        
        # Add information about standard error computation
        if self.bootstrap:
            smry.add_text(f"\nBootstrap: {self.bootstrap_reps} replications")
        elif self.compute_asymptotic_variance:
            smry.add_text(f"\nAsymptotic Variance: Corrected multivariate implementation")
            smry.add_text(f"Kernel type: {self.kernel_type}")
            smry.add_text(f"Density estimator: {self.model.density_estimator}")
        else:
            smry.add_text(f"\nStandard Errors: Not computed")
        
        return smry

    # Keep all other existing methods...
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
        """Calculate the estimator at specific point(s) X_points."""
        if not isinstance(X_points, np.ndarray):
            X_points = np.array(X_points)
        if X_points.ndim == 1:
            X_points = X_points.reshape(1, -1)
        
        num_points = X_points.shape[0]
        estimates = np.zeros(num_points)

        # Handle variable selection
        if variable_idx is None:
            if not self.interest:
                raise ValueError("No variable of interest specified or found.")
            variable_idx = self.interest[0]
        elif isinstance(variable_idx, str):
            if variable_idx not in self.model.exog_names:
                raise ValueError(f"Variable name '{variable_idx}' not in exog_names: {self.model.exog_names}")
            variable_idx = self.model.exog_names.index(variable_idx)

        # Get parametric coefficient and log_x for the specific variable_idx
        current_param_coef = self.parametric_results.params[variable_idx]
        try:
            idx_in_model_interest_list = self.model.interest.index(variable_idx)
            current_log_x_bool = self.model.log_x[idx_in_model_interest_list]
        except ValueError:
            warnings.warn(f"Variable index {variable_idx} not in model.interest. Defaulting log_x to False for it.")
            current_log_x_bool = False

        if variable_idx in self.model.binary_vars:
            X_0 = X_points.copy()
            X_1 = X_points.copy()
            X_0[:, variable_idx] = 0
            X_1[:, variable_idx] = 1
            
            m_0 = self.nonparam_model.predict(X_0)
            m_1 = self.nonparam_model.predict(X_1)
            
            m_0[m_0 == 0] = 1e-9
            estimates = np.exp(current_param_coef) * (m_1 / m_0) - 1
        else:
            m_at_points = self.nonparam_model.predict(X_points)
            m_prime_at_points = self.nonparam_model.derivative(X_points, variable_idx)

            m_at_points[m_at_points == 0] = 1e-9
            correction = m_prime_at_points / m_at_points
            
            if self.model.elasticity and current_log_x_bool:
                estimates = current_param_coef + correction
            elif not self.model.elasticity and not current_log_x_bool:
                estimates = current_param_coef + correction
            elif self.model.elasticity and not current_log_x_bool:
                estimates = current_param_coef + correction * X_points[:, variable_idx]
            else:
                x_val_col = X_points[:, variable_idx].copy()
                x_val_col[x_val_col == 0] = 1e-9
                estimates = current_param_coef + correction / x_val_col
        
        return estimates if num_points > 1 else estimates[0]

    def calculate_estimator_at_points(self, X_points, variable_idx=None):
        """Calculate estimator values at specific X points."""
        return self.estimate_at_point(X_points, variable_idx)

    def predict(self, exog=None):
        """Predict using the model."""
        if exog is None:
            exog = self.model.exog
            
        if self.parametric_method == 'ols':
            # Get the OLS predictions on the log scale
            log_pred = self.parametric_results.predict(exog)
            # Return transformed predictions (exponentiated and corrected)
            return np.exp(log_pred) * self.nonparam_model.predict(exog)
        else:  # PPML
            # Get the PPML predictions (already on the level scale)
            level_pred = self.parametric_results.fittedvalues
            # Return corrected predictions
            return level_pred * self.nonparam_model.predict(exog)

    def plot_distribution(self, variable_idx=None):
        """Plot the distribution of estimator values."""
        if isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        elif variable_idx is None:
            variable_idx = self.interest[0]
            
        if self.all_variables:
            i = list(self.interest).index(variable_idx)
            param_coef = self.parametric_coef[variable_idx]
            var_name = self.model.exog_names[variable_idx]
        else:
            i = self.interest.index(variable_idx)
            param_coef = self.parametric_coef[i]
            var_name = self.model.exog_names[variable_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.kdeplot(self.estimator_values[variable_idx], ax=ax)
        
        coef_name = "OLS" if self.parametric_method == 'ols' else "PPML"
        ax.axvline(x=param_coef, color='red', linestyle='--', label=f'{coef_name} Estimate')
        ax.axvline(x=np.mean(self.estimator_values[variable_idx]), color='blue', linestyle='--', 
                  label='Doubly Robust Average')
        ax.axvline(x=self.estimate_at_average(variable_idx), color='green', linestyle='--', 
                  label='Estimate at Average')
        
        estimator_name = "Estimator 1" if self.parametric_method == 'ols' else "Estimator 2"
        if self.model.elasticity:
            ax.set_title(f'Distribution of Elasticity Estimates for {var_name} ({estimator_name})')
            ax.set_xlabel('Elasticity')
        else:
            ax.set_title(f'Distribution of Semi-Elasticity Estimates for {var_name} ({estimator_name})')
            ax.set_xlabel('Semi-Elasticity')
        
        ax.set_ylabel('Density')
        ax.legend()
        
        return fig

    def plot_bootstrap_distribution(self, statistic='estimate_at_average', variable_idx=None):
        """
        Plot the bootstrap distribution for a given statistic.
        
        Parameters
        ----------
        statistic : str, default 'estimate_at_average'
            Which statistic to plot: 'ols_estimate', 'average_correction', 
            'average_estimate', or 'estimate_at_average'
        variable_idx : int or str, optional
            Variable for which to plot bootstrap distribution.
            If None, uses first variable of interest.
            
        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axis objects
        """
        if not self.bootstrap:
            raise ValueError("Bootstrap standard errors have not been computed. "
                           "Call fit() with bootstrap=True first.")
        
        # Handle variable selection
        if variable_idx is None:
            variable_idx = self.interest[0]
        elif isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
            variable_idx = self.model.exog_names.index(variable_idx)
        
        if variable_idx not in self.bootstrap_estimates_dict:
            raise ValueError(f"Variable {variable_idx} not found in bootstrap estimates.")
        
        # Map statistic names to column indices
        stat_map = {
            'ols_estimate': 0,
            'average_correction': 1,
            'average_estimate': 2,
            'estimate_at_average': 3
        }
        
        if statistic not in stat_map:
            raise ValueError(f"Unknown statistic '{statistic}'. "
                           f"Choose from: {list(stat_map.keys())}")
        
        col_idx = stat_map[statistic]
        bootstrap_values = self.bootstrap_estimates_dict[variable_idx][:, col_idx]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram and KDE
        sns.histplot(bootstrap_values, ax=ax, kde=True, alpha=0.7)
        
        # Add vertical lines for key statistics
        mean_val = np.mean(bootstrap_values)
        median_val = np.median(bootstrap_values)
        
        ax.axvline(x=mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        ax.axvline(x=median_val, color='blue', linestyle='--', label=f'Median: {median_val:.4f}')
        
        # Add confidence interval lines
        ci_lower = np.percentile(bootstrap_values, 2.5)
        ci_upper = np.percentile(bootstrap_values, 97.5)
        ax.axvline(x=ci_lower, color='green', linestyle=':', alpha=0.7, label='95% CI')
        ax.axvline(x=ci_upper, color='green', linestyle=':', alpha=0.7)
        
        # Formatting
        var_name = self.model.exog_names[variable_idx]
        ax.set_title(f'Bootstrap Distribution of {statistic.replace("_", " ").title()} for {var_name}')
        ax.set_xlabel(statistic.replace("_", " ").title())
        ax.set_ylabel('Density')
        ax.legend()
        
        return fig, ax
    
    def bootstrap_confidence_interval(self, alpha=0.05, method='percentile', variable_idx=None):
        """
        Calculate bootstrap confidence intervals.
        
        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for (1-α)×100% confidence interval
        method : str, default 'percentile'
            Method for calculating confidence intervals: 'percentile' or 'bias_corrected'
        variable_idx : int or str, optional
            Variable for which to compute confidence intervals.
            If None, computes for all variables of interest.
            
        Returns
        -------
        dict
            Dictionary with confidence intervals for each statistic and variable
        """
        if not self.bootstrap:
            raise ValueError("Bootstrap standard errors have not been computed. "
                           "Call fit() with bootstrap=True first.")
        
        # Handle variable selection
        if variable_idx is None:
            variables = list(self.bootstrap_estimates_dict.keys())
        else:
            if isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
                variable_idx = self.model.exog_names.index(variable_idx)
            variables = [variable_idx]
        
        confidence_intervals = {}
        
        stat_names = ['ols_estimate', 'average_correction', 'average_estimate', 'estimate_at_average']
        
        for var_idx in variables:
            var_name = self.model.exog_names[var_idx]
            confidence_intervals[var_name] = {}
            
            bootstrap_values = self.bootstrap_estimates_dict[var_idx]
            
            for i, stat_name in enumerate(stat_names):
                values = bootstrap_values[:, i]
                
                if method == 'percentile':
                    lower = np.percentile(values, 100 * alpha / 2)
                    upper = np.percentile(values, 100 * (1 - alpha / 2))
                elif method == 'bias_corrected':
                    # Bias-corrected and accelerated (BCa) bootstrap
                    # This is a simplified version - full BCa requires more computation
                    n_boot = len(values)
                    
                    # Bias correction
                    original_estimate = getattr(self, stat_name.replace('_estimate', '').replace('_', '_'))(var_idx) if hasattr(self, stat_name.replace('_estimate', '').replace('_', '_')) else np.mean(values)
                    bias_correction = stats.norm.ppf((np.sum(values < original_estimate)) / n_boot)
                    
                    # Acceleration (simplified - assumes jackknife acceleration = 0)
                    acceleration = 0
                    
                    # Adjusted percentiles
                    z_alpha_2 = stats.norm.ppf(alpha / 2)
                    z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
                    
                    alpha_1 = stats.norm.cdf(bias_correction + (bias_correction + z_alpha_2) / (1 - acceleration * (bias_correction + z_alpha_2)))
                    alpha_2 = stats.norm.cdf(bias_correction + (bias_correction + z_1_alpha_2) / (1 - acceleration * (bias_correction + z_1_alpha_2)))
                    
                    lower = np.percentile(values, 100 * alpha_1)
                    upper = np.percentile(values, 100 * alpha_2)
                else:
                    raise ValueError(f"Unknown method '{method}'. Use 'percentile' or 'bias_corrected'.")
                
                confidence_intervals[var_name][stat_name] = (lower, upper)
        
        return confidence_intervals
    
    def test_ppml(self):
        """Test the consistency of PPML estimation."""
        ppml_mod = sm.GLM(self.model.endog, self.model.exog, 
                          family=sm.families.Poisson()).fit(cov_type='HC3')
        # Assuming AssumptionTest is defined as imported
        from .ppml_consistency import AssumptionTest
        return AssumptionTest(ppml_mod).test_direct()

    def calculate_estimator_at_points_parallel(self, X_points, variable_idx=None, n_jobs=-1, backend='threading'):
        """Calculate estimator values with parallel processing (pickle-safe).
        
        Parameters
        ----------
        X_points : array-like
            Points at which to calculate the estimator
        variable_idx : int or str, optional
            Variable for which to calculate estimates. If None, uses first variable of interest.
        n_jobs : int, default -1
            Number of parallel jobs. -1 means use all processors.
        backend : str, default 'threading'
            Joblib backend to use: 'threading' or 'multiprocessing'
            
        Returns
        -------
        ndarray
            Array of estimator values at the specified points
        """
        import joblib
        
        if not isinstance(X_points, np.ndarray):
            X_points = np.array(X_points)
        if X_points.ndim == 1:
            X_points = X_points.reshape(1, -1)

        if backend == 'threading' and n_jobs == 1:
            return self.estimate_at_point(X_points, variable_idx)
        
        if backend == 'threading':
            estimator_values = joblib.Parallel(n_jobs=n_jobs, backend='threading')(
                joblib.delayed(lambda x_row: self.estimate_at_point(x_row.reshape(1, -1), variable_idx))(x_point_row) 
                for x_point_row in X_points
            )
        else:  # multiprocessing
            args = [(x_point_row, self, variable_idx) for x_point_row in X_points]
            estimator_values = joblib.Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                joblib.delayed(_estimate_single_point_worker)(arg) for arg in args
            )
        
        return np.array(estimator_values).flatten()

    def plot_correction(self, variable_idx=None):
        """Plot the correction term against the regressor of interest.
        
        Parameters
        ----------
        variable_idx : int or str, optional
            Variable for which to plot correction terms.
            If None, uses first variable of interest.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the correction term plot
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
        
        method_name = "OLS" if self.parametric_method == 'ols' else "PPML"
        ax.set_title(f'Correction Term vs {self.model.exog_names[variable_idx]} ({method_name}-based)')
        ax.set_xlabel(f'Regressor: {self.model.exog_names[variable_idx]}')
        ax.set_ylabel('Correction Term')
        
        return fig

    def std_error_at_average(self, variable_idx=None):
        """Calculate the standard error of the estimator at average values.
        
        Parameters
        ----------
        variable_idx : int or str, optional
            Variable for which to compute standard error.
            
        Returns
        -------
        float
            Standard error at average X values
            
        Raises
        ------
        ValueError
            If neither asymptotic variance nor bootstrap is available
        """
        if self.compute_asymptotic_variance:
            return self.asymptotic_standard_error_at_average(variable_idx)
        elif self.bootstrap:
            if hasattr(self, 'bootstrap_se_dict'):
                if isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
                    variable_idx = self.model.exog_names.index(variable_idx)
                elif variable_idx is None:
                    variable_idx = self.interest[0]
                return self.bootstrap_se_dict[variable_idx][3]  # Index 3 is for 'Estimate at Average'
            else:
                raise ValueError("Bootstrap standard errors not computed properly.")
        else:
            raise ValueError("Standard error computation requires asymptotic variance or bootstrap. "
                           "Set compute_asymptotic_variance=True or use bootstrap.")

    def conf_int(self, alpha=0.05, point=None, variable_idx=None):
        """Calculate confidence intervals for the estimator at a specific point.
        
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
            
        Raises
        ------
        ValueError
            If neither asymptotic variance nor bootstrap is available
        """
        if self.compute_asymptotic_variance:
            if point is None:
                point = np.mean(self.model.exog, axis=0).reshape(1, -1)
            return self.asymptotic_confidence_interval(point, variable_idx, alpha)
        elif self.bootstrap:
            # Use bootstrap confidence intervals
            if isinstance(variable_idx, str) and variable_idx in self.model.exog_names:
                variable_idx = self.model.exog_names.index(variable_idx)
            elif variable_idx is None:
                variable_idx = self.interest[0]
                
            var_name = self.model.exog_names[variable_idx]
            ci_dict = self.bootstrap_confidence_interval(alpha=alpha, variable_idx=variable_idx)
            return ci_dict[var_name]['estimate_at_average']
        else:
            raise ValueError("Confidence interval computation requires asymptotic variance or bootstrap. "
                           "Set compute_asymptotic_variance=True or use bootstrap.")

    def asymptotic_variance_matrix(self, x):
        """Compute full d×d asymptotic variance-covariance matrix at point x.
        
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
            
        where:
        - :math:`\\hat{\\sigma}_w^2(x)` is the conditional variance of :math:`e^u` given :math:`X=x`
        - :math:`\\hat{f}_X(x)` is the estimated density at :math:`x`
        - :math:`h` is the bandwidth
        - :math:`\\hat{m}(x)` is the estimated conditional expectation
        - :math:`\\nabla \\hat{m}(x)` is the gradient of :math:`\\hat{m}` at :math:`x`
        - :math:`\\mathcal{K}_0` and :math:`\\mathcal{K}_1` are kernel constants
        """
        if not self.compute_asymptotic_variance:
            raise ValueError("Asymptotic variance computation is disabled. "
                           "Set compute_asymptotic_variance=True to enable.")
        
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


class NNDensityEstimator:
    """
    Unified neural network density estimator for both density estimation and conditional variance.
    
    This class provides a unified interface for neural network-based density estimation
    that can be used for both f_X(x) estimation and conditional variance σ²_w(x) estimation.
    
    Parameters
    ----------
    X : array_like of shape (n, d)
        Input data for density estimation
    y : array_like of shape (n,), optional
        Target variable for conditional density/variance estimation.
        If None, estimates marginal density of X.
    target_type : str, default 'density'
        Type of target estimation: 'density' for f_X(x), 'conditional_variance' for σ²_w(x)
    hidden_layers : list, default [64, 32]
        List of hidden layer sizes
    activation : str, default 'relu'
        Activation function for hidden layers
    epochs : int, default 100
        Number of training epochs
    batch_size : int, default 32
        Batch size for training
    validation_split : float, default 0.2
        Fraction of data to use for validation
    verbose : int, default 1
        Verbosity level during training
    **kwargs : dict
        Additional parameters for neural network training
        
    Attributes
    ----------
    model : tensorflow.keras.Model
        Fitted neural network model
    target_type : str
        Type of target estimation
    is_fitted : bool
        Whether the model has been fitted
    """
    
    def __init__(self, X, y=None, target_type='density', hidden_layers=[64, 32], 
                 activation='relu', epochs=100, batch_size=32, validation_split=0.2,
                 verbose=1, **kwargs):
        """Initialize the neural network density estimator."""
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            raise ImportError("TensorFlow is required for neural network density estimator. "
                             "Please install tensorflow package.")
        
        self.X = np.array(X)
        self.y = np.array(y) if y is not None else None
        self.target_type = target_type.lower()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        
        # Store additional parameters
        self.kwargs = kwargs
        
        # Initialize model attributes
        self.model = None
        self.is_fitted = False
        self.mean = None
        self.std = None
        self.scaling_factor = None
        
        # Fit the model upon initialization
        self._fit()
    
    def _fit(self):
        """Fit the neural network model."""
        if self.target_type == 'density':
            self._fit_density_model()
        elif self.target_type == 'conditional_variance':
            self._fit_conditional_variance_model()
        else:
            raise ValueError(f"Unknown target_type '{self.target_type}'. "
                           f"Use 'density' or 'conditional_variance'.")
        
        self.is_fitted = True
    
    def _fit_density_model(self):
        """Fit neural network for density estimation f_X(x)."""
        # For density estimation, we use a variational approach
        # This is a simplified implementation - a full implementation would use 
        # normalizing flows or mixture density networks
        
        # Create synthetic targets for density estimation using KDE
        from statsmodels.nonparametric.kernel_density import KDEMultivariate
        
        var_type = 'c' * self.X.shape[1]
        kde = KDEMultivariate(data=self.X, var_type=var_type, bw='normal_reference')
        
        # Evaluate KDE at training points to create targets
        targets = np.array([kde.pdf(x) for x in self.X])
        
        # Normalize targets for numerical stability
        self.mean = np.mean(targets)
        self.std = np.std(targets)
        if self.std < 1e-8:
            self.std = 1.0
        
        normalized_targets = (targets - self.mean) / self.std
        
        # Create and compile model
        self.model = self._create_model('density')
        
        # Convert to tensors
        X_tensor = self.tf.convert_to_tensor(self.X, dtype=self.tf.float32)
        y_tensor = self.tf.convert_to_tensor(normalized_targets, dtype=self.tf.float32)
        
        # Set up callbacks
        callbacks = [
            self.tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.kwargs.get('patience', 10),
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Fit the model
        self.model.fit(
            x=X_tensor,
            y=y_tensor,
            validation_split=self.validation_split,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=self.verbose
        )
    
    def _fit_conditional_variance_model(self):
        """Fit neural network for conditional variance estimation."""
        if self.y is None:
            raise ValueError("Target variable y is required for conditional variance estimation.")
        
        # Normalize targets for numerical stability
        min_y = np.min(self.y)
        if min_y <= 0:
            min_y = 1e-10
        
        self.scaling_factor = min_y
        normalized_y = self.y / self.scaling_factor
        
        self.mean = np.mean(normalized_y)
        self.std = np.std(normalized_y)
        if self.std < 1e-8:
            self.std = 1.0
        
        scaled_y = (normalized_y - self.mean) / self.std
        
        # Create and compile model
        self.model = self._create_model('conditional_variance')
        
        # Convert to tensors
        X_tensor = self.tf.convert_to_tensor(self.X, dtype=self.tf.float32)
        y_tensor = self.tf.convert_to_tensor(scaled_y, dtype=self.tf.float32)
        
        # Set up callbacks with regularization for variance estimation
        callbacks = [
            self.tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.kwargs.get('patience', 10),
                restore_best_weights=True,
                verbose=0
            ),
            self.tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(5, self.kwargs.get('patience', 10) // 2),
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        # Fit the model
        try:
            history = self.model.fit(
                x=X_tensor,
                y=y_tensor,
                validation_split=self.validation_split,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=self.verbose
            )
            
            # Check if training was successful
            final_loss = history.history['loss'][-1]
            if np.isnan(final_loss) or np.isinf(final_loss):
                raise ValueError("Neural network training failed - loss is NaN or infinite")
                
        except Exception as e:
            raise RuntimeError(f"Neural network conditional variance training failed: {e}")
    
    def _create_model(self, model_type):
        """Create neural network model based on type."""
        # Create the model architecture
        model = self.tf.keras.Sequential()
        model.add(self.tf.keras.layers.Input(shape=(self.X.shape[1],)))
        
        # Add hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(self.tf.keras.layers.Dense(
                units, 
                activation=self.activation,
                kernel_regularizer=self.tf.keras.regularizers.l2(1e-4),
                kernel_initializer='he_normal'
            ))
            
            # Add batch normalization for stability
            model.add(self.tf.keras.layers.BatchNormalization())
            
            # Add dropout for regularization (except last hidden layer)
            if i < len(self.hidden_layers) - 1:
                model.add(self.tf.keras.layers.Dropout(0.1))
        
        # Output layer
        if model_type == 'density':
            # For density estimation, use positive output
            model.add(self.tf.keras.layers.Dense(1, activation='softplus'))
        else:
            # For conditional variance, no activation (will enforce non-negativity in prediction)
            model.add(self.tf.keras.layers.Dense(1))
        
        # Compile with appropriate settings
        optimizer = self.tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def predict_density(self, X):
        """
        Predict density or conditional variance at given points.
        
        Parameters
        ----------
        X : array_like of shape (n, d)
            Points at which to evaluate the density/variance
            
        Returns
        -------
        ndarray of shape (n,)
            Predicted density or conditional variance values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        X = np.atleast_2d(X)
        X_tensor = self.tf.convert_to_tensor(X, dtype=self.tf.float32)
        
        # Get predictions from neural network
        predictions = self.model.predict(X_tensor, verbose=0)
        
        if self.target_type == 'density':
            # Denormalize density predictions
            unstandardized = predictions.reshape(-1) * self.std + self.mean
            # Ensure positive density values
            return np.maximum(unstandardized, 1e-10)
            
        elif self.target_type == 'conditional_variance':
            # Denormalize conditional variance predictions
            unstandardized = predictions.reshape(-1) * self.std + self.mean
            # Unscale back to original scale
            unscaled = unstandardized * self.scaling_factor
            # Ensure non-negative variance values
            return np.maximum(unscaled, 0.0)
    
    def fit(self, X):
        """
        Interface compatibility method for conditional variance estimation.
        
        This method provides the same interface as KernelReg.fit() for seamless
        integration with existing conditional variance code.
        
        Parameters
        ----------
        X : array_like of shape (n, d)
            Points at which to evaluate
            
        Returns
        -------
        tuple
            (predictions, marginal_effects) where:
            - predictions: ndarray of shape (n,) with predicted values
            - marginal_effects: None (for compatibility)
        """
        if self.target_type != 'conditional_variance':
            raise ValueError("fit() method is only available for conditional_variance target_type.")
        
        predictions = self.predict_density(X)
        return (predictions, None)


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
    """Fixed Neural network regression model for nonparametric estimation."""
    
    def __init__(self, model, mean, std, scaling_factor, binary_vars, ordinal_vars):
        """Initialize the neural network regression model."""
        super(NNRegressionModel, self).__init__(binary_vars, ordinal_vars)
        self.model = model
        self.mean = mean
        self.std = std
        self.scaling_factor = scaling_factor
        self.bandwidth = None
        
        # Import tensorflow and set device more robustly
        import tensorflow as tf
        self.tf = tf
        
        # More robust device detection
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Set memory growth to avoid allocation issues
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.use_gpu = True
                print(f"✅ Using GPU: {len(gpus)} GPU(s) available")
            else:
                self.use_gpu = False
                print("💻 Using CPU for computations")
        except Exception as e:
            self.use_gpu = False
            print(f"⚠️ GPU setup failed, using CPU: {e}")
    
    def predict(self, X):
        """Predict the expected value m(x) for given data."""
        try:
            # Convert to tensor with proper device placement
            if self.use_gpu:
                with self.tf.device('/GPU:0'):
                    X_tensor = self.tf.convert_to_tensor(X, dtype=self.tf.float32)
                    predictions = self.model(X_tensor, training=False)
            else:
                X_tensor = self.tf.convert_to_tensor(X, dtype=self.tf.float32)
                predictions = self.model(X_tensor, training=False)
            
            predictions_np = predictions.numpy()
            # Unstandardize and unscale to get back to original scale
            return (predictions_np.reshape(-1) * self.std + self.mean) * self.scaling_factor
            
        except Exception as e:
            warnings.warn(f"Neural network prediction failed: {e}. Using fallback.")
            # Fallback to simple prediction
            return np.ones(len(X)) * self.scaling_factor
    
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
            return self._ordinal_derivative(X, index)
        
        # For continuous variables, use automatic differentiation with proper error handling
        else:
            return self._continuous_derivative(X, index)
    
    def _ordinal_derivative(self, X, index):
        """Handle derivative calculation for ordinal variables."""
        unique_values = np.sort(np.unique(X[:, index]))
        
        if len(unique_values) <= 1:
            return np.zeros(len(X))
            
        if len(X) == 1:  # Single point
            current_value = X[0, index]
            idx = np.searchsorted(unique_values, current_value)
            
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
            # Multiple points
            derivatives = np.zeros(len(X))
            for i in range(len(X)):
                x_i = X[i:i+1]
                derivatives[i] = self.derivative(x_i, index)[0]
            return derivatives
    
    def _continuous_derivative(self, X, index):
        """Handle derivative calculation for continuous variables with robust automatic differentiation."""
        try:
            # Method 1: Try automatic differentiation with proper device handling
            if self.use_gpu:
                with self.tf.device('/GPU:0'):
                    return self._compute_gradient_gpu(X, index)
            else:
                return self._compute_gradient_cpu(X, index)
                
        except Exception as e:
            warnings.warn(f"Automatic differentiation failed: {e}. Using numerical differentiation.")
            return self._numerical_derivative(X, index)
    
    def _compute_gradient_gpu(self, X, index):
        """Compute gradient using GPU with proper error handling."""
        X_tensor = self.tf.convert_to_tensor(X, dtype=self.tf.float32)
        
        with self.tf.GradientTape(persistent=False) as tape:
            # Create Variable without device attribute access - THIS IS THE FIX
            X_var = self.tf.Variable(X_tensor, trainable=True)
            tape.watch(X_var)
            
            # Compute predictions
            raw_pred = self.model(X_var, training=False)
            # Apply scaling and standardization
            predictions = raw_pred * self.std * self.scaling_factor + self.mean * self.scaling_factor
            
        # Compute gradients
        gradients = tape.gradient(predictions, X_var)
        
        if gradients is None:
            raise ValueError("Gradient computation returned None")
            
        return gradients[:, index].numpy()
    
    def _compute_gradient_cpu(self, X, index):
        """Compute gradient using CPU."""
        X_tensor = self.tf.convert_to_tensor(X, dtype=self.tf.float32)
        
        with self.tf.GradientTape(persistent=False) as tape:
            # Create Variable without accessing device attribute - THIS IS THE FIX
            X_var = self.tf.Variable(X_tensor, trainable=True)
            tape.watch(X_var)
            
            # Compute predictions
            raw_pred = self.model(X_var, training=False)
            # Apply scaling and standardization  
            predictions = raw_pred * self.std * self.scaling_factor + self.mean * self.scaling_factor
            
        # Compute gradients
        gradients = tape.gradient(predictions, X_var)
        
        if gradients is None:
            raise ValueError("Gradient computation returned None")
            
        return gradients[:, index].numpy()
    
    def _numerical_derivative(self, X, index):
        """Fallback numerical differentiation method."""
        epsilon = 1e-5
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, index] += epsilon
        X_minus[:, index] -= epsilon
        
        pred_plus = self.predict(X_plus)
        pred_minus = self.predict(X_minus)
        
        return (pred_plus - pred_minus) / (2 * epsilon)

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
        """Calculate the derivative of m(x) with respect to a specific regressor using analytical derivatives."""
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
        
        # For continuous variables, use analytical derivative of the polynomial
        else:
            # Ensure X is 2D
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Get the polynomial powers matrix and model coefficients
            powers = self.poly.powers_  # Shape: (n_features, n_input_features)
            coeffs = self.model.params  # Shape: (n_features,)
            
            # Initialize derivative values
            derivatives = np.zeros(X.shape[0])
            
            # Loop through each polynomial feature and compute its derivative
            for i, power_vec in enumerate(powers):
                # Get the power of the variable of interest in this feature
                power_of_interest = power_vec[index]
                
                # If the power is 0, this feature doesn't depend on the variable of interest
                # so its derivative is 0
                if power_of_interest == 0:
                    continue
                    
                # Calculate the derivative coefficient for this feature
                # d/dx_j [c * x_j^k * other_terms] = c * k * x_j^(k-1) * other_terms
                deriv_coeff = power_of_interest * coeffs[i]
                
                # Calculate the value of this derivative term at each X point
                feature_deriv = deriv_coeff * np.ones(X.shape[0])
                
                # Multiply by the appropriate powers of all variables
                for j, power in enumerate(power_vec):
                    if j == index:
                        # For the variable of interest, use power - 1
                        if power > 1:
                            feature_deriv *= X[:, j] ** (power - 1)
                        # If power == 1, x^(1-1) = x^0 = 1, so no multiplication needed
                    else:
                        # For other variables, use the original power
                        if power > 0:
                            feature_deriv *= X[:, j] ** power
                
                derivatives += feature_deriv
            
            # Scale back to original scale (same as in predict method)
            return derivatives * self.scaling_factor


# Alias names for easier use
DRE = DoublyRobustElasticityEstimator
DREER = DoublyRobustElasticityEstimatorResults


# Example usage
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Generate sample data
    np.random.seed(6699)
    n = 5000
    x1 = np.random.normal(5, 1, n)
    x2 = x1 ** np.random.normal(1,1,n)
    
    # Create an exponential model
    y = np.exp(0.5 + 1.2 * x1 + np.random.uniform(-1.5*x1**2, 1.5*x1**2, n))
    
    df = pd.DataFrame({'y': y, 'x1': x1})
    
    # Compare both estimators
    print("="*60)
    print("ESTIMATOR 1 (OLS-based)")
    print("="*60)
    
    dre1 = DRE(df['y'], df[['x1']], interest='x1', estimator_type='ols', 
               kernel_params={'degree': 2})
    results1 = dre1.fit(method='ols', compute_asymptotic_variance=True)
    print(results1.summary())
    
    print("\n" + "="*60)
    print("ESTIMATOR 2 (PPML-based)")
    print("="*60)
    
    dre2 = DRE(df['y'], df[['x1']], interest='x1', estimator_type='ols', 
               kernel_params={'degree': 2})
    results2 = dre2.fit(method='ppml', compute_asymptotic_variance=True)
    print(results2.summary())
    
    # Plot distributions for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Estimator 1 distribution
    ax1.hist(results1.estimator_values[0], bins=50, alpha=0.7, density=True, label='Estimator 1')
    ax1.axvline(x=results1.estimate_at_average(0), color='red', linestyle='--', 
               label=f'Est. at Avg: {results1.estimate_at_average(0):.3f}')
    ax1.set_title('Estimator 1 (OLS-based)')
    ax1.set_xlabel('Semi-Elasticity')
    ax1.set_ylabel('Density')
    ax1.legend()
    
    # Estimator 2 distribution
    ax2.hist(results2.estimator_values[0], bins=50, alpha=0.7, density=True, label='Estimator 2')
    ax2.axvline(x=results2.estimate_at_average(0), color='red', linestyle='--', 
               label=f'Est. at Avg: {results2.estimate_at_average(0):.3f}')
    ax2.set_title('Estimator 2 (PPML-based)')
    ax2.set_xlabel('Semi-Elasticity')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Estimator 1 (OLS-based) estimate at average: {results1.estimate_at_average(0):.4f}")
    print(f"Estimator 2 (PPML-based) estimate at average: {results2.estimate_at_average(0):.4f}")
    print(f"Standard OLS estimate: {results1.parametric_results.params[0]:.4f}")
    print(f"Standard PPML estimate: {results2.parametric_results.params[0]:.4f}")


from dataclasses import dataclass

# You can add this class to the top of correction_estimator.py
@dataclass
class LeanBootstrapResult:
    """Stores the essential components of a bootstrap replication fit."""
    parametric_params: np.ndarray
    nonparam_model: object # This will hold the KernelRegressionModel, NNRegressionModel, etc.
    model_interest_indices: list
    model_log_x_flags: list
    model_binary_vars: np.ndarray
