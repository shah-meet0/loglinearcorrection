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
        hasconst: bool = True, # Do we need this?
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
        hasconst : bool, default True
            Whether the model includes a constant term.
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
        endog_names : str or None
            Name of dependent variable if available from input.
        exog_names : list of str or None
            Names of independent variables if available from input.
        endog_demeaned : ndarray
            Dependent variable after fixed effects transformation.
        exog_demeaned : ndarray
            Independent variables after fixed effects transformation.
        variable_types : dict
            Mapping of variable indices/names to detected types ('continuous',
            'binary', 'ordinal') for non-fixed-effect variables.
        fixed_effects : list or None
            Fixed effects specification.
        interest : list or None
            Variables of interest specification.
        hasconst : bool
            Whether model includes constant.
        k_constant : int
            Number of constants (1 if hasconst else 0).
        nobs : int
            Number of observations.

        Raises
        ------
        ValueError
            If endog, exog, or weights contain non-numeric data.

        Notes
        -----
        The initialization performs the following operations:

        1. Extracts and stores variable names from pandas objects if provided
        2. Converts all inputs to numpy arrays and validates numeric types
        3. Detects variable types (continuous, binary, ordinal) for non-fixed-
           effect variables using :func:`_detect_variable_types`
        4. Applies within-group demeaning for fixed effects using
           :func:`_apply_fixed_effects`
        5. Stores both original and transformed data for estimation
        """
        # Store specifications
        self.fixed_effects = fixed_effects
        self.interest = interest
        self.hasconst = hasconst
        self.k_constant = int(hasconst)

        # Extract names from pandas objects
        self.endog_names = (
            endog.name if isinstance(endog, pd.Series)
            else endog.columns[0] if isinstance(endog, pd.DataFrame)
            else None #probably should call them x1 to xn
        )
        self.exog_names = (
            exog.columns.tolist() if isinstance(exog, pd.DataFrame)
            else [exog.name] if isinstance(exog, pd.Series)
            else None #probably should call them x1 to xn
        )

        # Convert to arrays and validate numeric types
        endog_arr = np.asarray(endog)
        exog_arr = np.asarray(exog)
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

        # Apply fixed effects transformation
        self.endog_demeaned, self.exog_demeaned = _apply_fixed_effects(
            self.endog, self.exog, fixed_effects, self.exog_names
        )
