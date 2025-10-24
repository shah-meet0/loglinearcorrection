import numpy as np
import numpy.typing as npt
import pandas as pd

from .utils import _apply_fixed_effects, _detect_variable_types


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
