from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import KFold
import statsmodels.api as sm

from .utils import _apply_fixed_effects, _detect_variable_types, _delete_redundant, _initialize_fixed_effects, _adjust_names_after_deletion, _adjust_indices_after_deletion, _redundant_columns
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
        4. Stores both original data for estimation

        Variable type detection is important for proper handling in elasticity
        estimation. Users should carefully specify ordinal variables based on
        their domain knowledge, as automatic detection can be unreliable. Binary
        detection takes precedence to ensure proper handling of dichotomous
        variables.
        """
        # Store specifications
        self.ordinal = ordinal

        # Extract names from pandas objects or generate defaults
        self.endog_names = (
            endog.name if isinstance(endog, pd.Series) and endog.name is not None
            else endog.columns[0] if isinstance(endog, pd.DataFrame) and endog.columns[0] != 0
            else "y"
        )

        self._original_is_pandas = isinstance(exog, pd.DataFrame)

        # Convert to array first to get shape for default naming
        exog_arr = np.asarray(exog)
        n_vars = exog_arr.shape[1] if exog_arr.ndim == 2 else 1

        self.exog_names = (
            exog.columns.tolist() if isinstance(exog, pd.DataFrame)
            else [exog.name] if isinstance(exog, pd.Series)
            else [f"x{i + 1}" for i in range(n_vars)]
        )

        # Convert to arrays and validate numeric types
        endog_arr = np.asarray(endog)
        weights_arr = np.asarray(weights) if weights is not None else None

        # Store original data
        self.endog = endog_arr.ravel()
        self.exog = exog_arr if exog_arr.ndim == 2 else exog_arr.reshape(-1, 1)
        self.weights = weights_arr
        self.nobs = len(self.endog)

        # Identify non-fixed-effect variable indices
        if fixed_effects is None:
            non_fe_indices = list(range(self.exog.shape[1]))
            self.fe_indices = []
            self.fixed_effects = None
            self.fe_cols = None
        else:
            if self.exog_names and isinstance(fixed_effects[0], str):
                fe_indices = [self.exog_names.index(name) for name in fixed_effects]
            else:
                fe_indices = list(set(fixed_effects))

            self.fe_indices = fe_indices
            self.fe_cols = self.exog[:, fe_indices]
            self.fixed_effects = _initialize_fixed_effects(self.exog[:, fe_indices])
            non_fe_indices = [i for i in range(self.exog.shape[1]) if i not in fe_indices]

        # Adjust interest based on fe_indices
        self.exog = self.exog[:, non_fe_indices].astype(np.float64)
        adjusted_exog_names: list[str] = [self.exog_names[i] for i in non_fe_indices]
        self.exog_names: list[str] = adjusted_exog_names
        self.interest = self._parse_interest(interest, non_fe_indices)

        if not np.issubdtype(self.endog.dtype, np.number):
            raise ValueError("endog must contain only numeric data")
        if not np.issubdtype(self.exog.dtype, np.number):
            raise ValueError("exog must contain only numeric data")
        if weights_arr is not None and not np.issubdtype(self.weights.dtype, np.number):
            raise ValueError("weights must contain only numeric data")

        # FIGURE OUT REDUNDANT INDICES, PASS PARAMS TO FIT METHODS
        if self.fixed_effects is not None:
            endog_demeaned, exog_demeaned = _apply_fixed_effects(np.log(self.endog), self.exog, self.fixed_effects)
            redundant_idx = _redundant_columns(exog_demeaned)
            print("Redundant columns after applying fixed effects:", [self.exog_names[i] for i in redundant_idx])
            self.exog = _delete_redundant(self.exog, redundant_idx)
            exog_demeaned = _delete_redundant(exog_demeaned, redundant_idx)
            self.exog_names = _adjust_names_after_deletion(self.exog_names, redundant_idx)
            self.interest = _adjust_indices_after_deletion(self.interest, redundant_idx)
            self.temp_exog_demeaned = exog_demeaned  # store for later use in fit
            self.ols_res = sm.WLS(endog_demeaned, exog_demeaned, weights=self.weights if self.weights else 1).fit()
            self.beta = self.ols_res.params
        else:
            self.ols_res = sm.WLS(np.log(self.endog), self.exog, weights=self.weights if self.weights else 1).fit()
            self.beta = self.ols_res.params

        # Detect variable types for interest variables
        self.variable_types = _detect_variable_types(
            self.exog, self.interest
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

    def _process_fold(
            self,
            train_idx: np.ndarray,
            test_idx: np.ndarray,
            weight_fold: np.ndarray,
            interest_indices: list[int],
            m_params: dict,
            density_params: dict
    ) -> dict:
        """
        Process a single fold of cross-fitting.

        Returns
        -------
        dict
            Contains: beta, m_results, f_results, moments, alpha_x, theta_x
        """
        # Apply fixed effects if needed
        if self.fixed_effects is not None:
            endog_demeaned, exog_demeaned = _apply_fixed_effects(
                np.log(self.endog), self.exog,
                algorithm=self.fixed_effects,
                weights=weight_fold.reshape(-1, 1)
            )
            exog_ols_train = exog_demeaned[train_idx]
            exog_ols_test = exog_demeaned[test_idx]
            log_endog_ols_train = endog_demeaned[train_idx]
            log_endog_ols_test = endog_demeaned[test_idx]
        else:
            exog_ols_train = self.exog[train_idx]
            exog_ols_test = self.exog[test_idx]
            log_endog_ols_train = np.log(self.endog[train_idx])
            log_endog_ols_test = np.log(self.endog[test_idx])

        exog_train = self.exog[train_idx]
        exog_test = self.exog[test_idx]
        weights_train = weight_fold[train_idx] if self.weights is not None else None

        # Step 1: Estimate OLS coefficients
        ols_model = sm.WLS(log_endog_ols_train, exog_ols_train,
                           weights=weights_train) if weights_train is not None else sm.OLS(log_endog_ols_train,
                                                                                           exog_ols_train)
        ols_results = ols_model.fit()
        beta = ols_results.params

        # Compute residuals
        log_residuals_test = log_endog_ols_test - exog_ols_test @ beta
        exp_residuals_test = np.exp(log_residuals_test)
        log_residuals_train = ols_results.resid
        exp_residuals_train = np.exp(log_residuals_train)

        # Step 2: Estimate nuisance functions
        m_model = NNModelNuisance(variable_types=self.variable_types, **m_params['arch_params'])
        m_results = m_model.fit(exog_train, exp_residuals_train, **m_params['fit_params'])

        m_test, m_prime_test = m_results.derivative(exog_test, interest_indices)
        if any(m_test <= 0):
            raise ValueError("Predicted m(x) has non-positive values")
        p_test = exp_residuals_test - m_test
        real_resid_fold = np.exp(self.ols_res.resid[test_idx])

        # Estimate density
        density_model = NNModelDensity(variable_types=self.variable_types, **density_params['arch_params'])
        f_results = density_model.fit(exog_train, interest=interest_indices, **density_params['fit_params'])
        alpha_weights = f_results.alpha_weight(exog_test, interest_indices)

        # Step 3: Construct moments
        n_interest = len(interest_indices)
        fold_moments = np.zeros((len(test_idx), n_interest))
        fold_derivative = np.zeros((len(test_idx), n_interest,
                                    2 * n_interest))  # First len(interest) for elasticities, second elasiticities w.r.t beta
        identity = np.eye(n_interest)
        fold_derivative[:, :n_interest, :n_interest] = -1 * identity  # fill in ones on interest diagonal
        fold_alpha_x = []
        fold_theta_x = []

        for moment_idx, var_idx in enumerate(interest_indices):
            var_type = self.variable_types.get(var_idx, 'continuous')

            if var_type == 'continuous':
                alpha_weight = alpha_weights[:, moment_idx]
                alpha_test = -alpha_weight / m_test
                m_semi_elast_test = m_prime_test[:, moment_idx] / m_test
                theta_test = self.beta[var_idx] + m_semi_elast_test

                fold_derivative[:, moment_idx, n_interest:] = identity[moment_idx, :] - (alpha_test * real_resid_fold)[
                                                                                        :, None] * exog_test[:,
                                                                                                   interest_indices]

            elif var_type == 'binary':
                # Binary variable logic (keeping existing implementation)
                exog_test_flip = exog_test.copy()
                exog_test_flip[:, var_idx] = 1 - exog_test_flip[:, var_idx]
                m_shifted_test = m_results.predict(exog_test_flip)
                probability_var = alpha_weights[:, moment_idx]

                alpha_0 = (1 - exog_test[:, var_idx].astype(np.int64)) * m_shifted_test / (
                            m_test ** 2 * probability_var)
                alpha_1 = (exog_test[:, var_idx].astype(np.int64)) / (probability_var * m_shifted_test)
                correction_0 = (1 - exog_test[:, var_idx].astype(np.int64)) * m_shifted_test / m_test
                correction_1 = (exog_test[:, var_idx].astype(np.int64)) * m_test / m_shifted_test

                theta_test = np.exp(self.beta[var_idx]) * (correction_1 + correction_0) - 1
                alpha_test = np.exp(self.beta[var_idx]) * (alpha_1 - alpha_0)

                derivative_leading_term = np.zeros(shape=(len(test_idx), n_interest))
                derivative_leading_term[:, moment_idx] = theta_test + 1 + alpha_test * (real_resid_fold - m_test)
                derivative_second_term = (-1 * alpha_test * real_resid_fold)[:, None] * exog_test[:, interest_indices]
                fold_derivative[:, moment_idx, n_interest:] = derivative_leading_term + derivative_second_term
            else:  # ordinal
                theta_test = 0
                alpha_test = 0

            fold_moments[:, moment_idx] = theta_test + alpha_test * p_test
            fold_alpha_x.append(alpha_test)
            fold_theta_x.append(theta_test)

        return {
            'beta': beta,
            'm_results': m_results,
            'f_results': f_results,
            'moments': fold_moments,
            'derivative': fold_derivative,
            'alpha_x': np.column_stack(fold_alpha_x) if fold_alpha_x else np.array([]),
            'theta_x': np.column_stack(fold_theta_x) if fold_theta_x else np.array([]),
            'test_idx': test_idx
        }

    def fit(self, n_folds: int = 5, random_state: Optional[int] = None,
            m_params: Dict = None, density_params: Dict = None, fit_ppml=False) -> 'DREEMR':
        """Fit the Doubly Robust Nonparametric Orthogonal elasticity estimator."""

        m_params = self._parse_m_params(m_params)
        density_params = self._parse_density_params(density_params)

        # Determine interest indices
        if self.interest is None:
            interest_indices = list(range(self.exog.shape[1]))
        else:
            interest_indices = list(self.interest)

        # Set up cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # Process each fold
        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.exog)):
            weight_fold = np.ones(self.nobs)
            weight_fold[test_idx] = 1e-10  # temporary hack to prevent fe from dying on zero weights
            if self.weights is not None:
                weight_fold = weight_fold * self.weights

            print(f'Processing fold {fold_idx + 1}/{n_folds}')
            fold_result = self._process_fold(
                train_idx, test_idx, weight_fold,
                interest_indices, m_params, density_params
            )
            fold_results.append(fold_result)

        # Aggregate moments across folds
        n_interest = len(interest_indices)
        n_params = (
                               2 + fit_ppml) * n_interest  # Elasticities for interest  + OLS params for interest + PPML params if applicable
        moments = np.zeros((self.nobs, n_params))
        derivative = np.zeros(shape=(self.nobs, n_params, n_params))
        fold_weights = np.zeros(self.nobs)

        for fold in fold_results:
            moments[fold['test_idx'], :n_interest] = fold['moments']
            fold_weights[fold['test_idx']] = self.weights[fold['test_idx']] if self.weights is not None else 1.0
            derivative[fold['test_idx'], :n_interest, :2 * n_interest] = fold['derivative']

        ols_moments, ols_derivative = self._process_ols(interest_indices)
        moments[:, n_interest: 2 * n_interest] = ols_moments
        derivative[:, n_interest: 2 * n_interest, n_interest:2 * n_interest] = ols_derivative

        if fit_ppml:
            ppml_moments, ppml_derivative, ppml_params = self._process_ppml(interest_indices)
            moments[:, 2 * n_interest:] = ppml_moments
            derivative[:, 2 * n_interest:, 2 * n_interest:] = ppml_derivative

        # Extract point estimates
        moment_means = np.average(moments, axis=0, weights=fold_weights)
        elasticity_estimates = moment_means[:len(interest_indices)]

        # Prepare data based on input type
        is_pandas = hasattr(self, '_original_is_pandas') and self._original_is_pandas

        if is_pandas:
            import pandas as pd
            elasticities_df = pd.DataFrame({
                'variable': [self.exog_names[i] for i in interest_indices],
                'estimate': elasticity_estimates,
                'type': [self.variable_types.get(i, 'continuous') for i in interest_indices],
                'index': interest_indices
            }).set_index('variable')
        else:
            elasticities_df = elasticity_estimates

        # Create results object
        results = DREEMR(
            elasticities=elasticities_df,
            beta=self.beta[interest_indices],
            ols_results=self.ols_res,
            exog_names=self.exog_names,
            endog_name=self.endog_names,
            n_folds=n_folds,
            nobs=self.nobs,
            variable_types=self.variable_types,
            interest_indices=interest_indices,
            fold_results=fold_results,
            moments=moments,
            derivative=derivative,
            fold_weights=fold_weights,
            gamma=ppml_params if fit_ppml else None
        )

        # Compute variances
        results.compute_variances()

        return results

    def _parse_m_params(self, m_params):
        if m_params is None:
            m_params = {'arch_params': {'hidden_layers': [512, 512, 512],
                                        'input_size': 0,
                                        'output_size': 0,
                                        'output_activation': 'identity'
                                        },
                        'fit_params': {}}

        for key in m_params.keys():
            if key not in ['arch_params', 'fit_params']:
                raise ValueError(f"Invalid key in m_params: {key} . Must be 'arch_params' or 'fit_params'.")

        if 'fit_params' not in m_params:
            m_params['fit_params'] = {}

        if 'arch_params' not in m_params:
            m_params['arch_params'] = {'hidden_layers': [512, 512, 512],
                                       'input_size': 0,
                                       'output_size': 0,
                                       'output_activation': 'identity'}

        m_params['arch_params']['input_size'] = self.exog.shape[1]
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

    def _parse_interest(self, interest, non_fe_indices):
        if interest is None:
            interest = [i for i in range(len(non_fe_indices))]
        else:
            if isinstance(interest, int) or isinstance(interest, str):
                interest = [interest]
            elif not isinstance(interest, list):
                raise ValueError("interest must be an int, str, or list of int/str")

            if self.exog_names and isinstance(interest[0], str):
                interest = [self.exog_names.index(name) for name in interest if name in self.exog_names]
            interest = [non_fe_indices.index(i) for i in interest if i in non_fe_indices]

        return interest

    def _process_ols(self, interest_indices: list[int]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        resid = self.ols_res.resid
        relevant_exog = self.exog[:, interest_indices]
        if self.fixed_effects is not None:
            relevant_exog = self.temp_exog_demeaned[:, interest_indices]
        moments_ols = resid[:, None] * relevant_exog
        derivatives_ols = -1 * relevant_exog[:, :, None] * relevant_exog[:, None, :]
        return moments_ols, derivatives_ols

    def _process_ppml(self, interest_indices: list[int]) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        import pyfixest
        df = pd.DataFrame(self.exog, columns=self.exog_names)
        df[self.endog_names] = self.endog
        fe_names = [f'F{i}' for i in range(self.fe_cols.shape[1])] if getattr(self, "fe_cols", None) is not None else []

        if fe_names:
            fe_df = pd.DataFrame(self.fe_cols, columns=fe_names)
            df = pd.concat([df, fe_df], axis=1)

        exog_part = " + ".join(self.exog_names) if self.exog_names else "1"
        fe_part = " + ".join(fe_names)
        # PPML with FEs typically drops the intercept to avoid collinearity with FEs
        formula = f"{self.endog_names} ~ {exog_part}" + (f" | {fe_part}" if fe_part else "")

        ppml_model = pyfixest.fepois(formula, data=df, drop_intercept=True)  # Weights not supported
        resid = ppml_model.resid()
        fitted_values = self.endog - resid
        ppml_moments = resid[:, None] * self.exog[:, interest_indices]
        ppml_derivatives = -1 * fitted_values[:, None, None] * (
                    self.exog[:, interest_indices, None] * self.exog[:, None, interest_indices])
        ppml_params = ppml_model.coef().loc[[self.exog_names[i] for i in interest_indices]].values
        return ppml_moments, ppml_derivatives, ppml_params


DREEM = DoublyRobustElasticityEstimatorModel
