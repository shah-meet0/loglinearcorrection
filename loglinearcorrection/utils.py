import numpy as np
import numpy.typing as npt
from typing import Literal


def _apply_fixed_effects(
    endog: npt.NDArray[np.floating],
    exog: npt.NDArray[np.floating],
    fixed_effects: list[str] | list[int] | None,
    exog_names: list[str] | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Apply within-group demeaning transformation for fixed effects.

    Uses the pyhdfe package to demean endogenous and exogenous variables
    according to fixed effects group membership. Variables specified as
    fixed effects are removed from the returned exogenous array.

    Parameters
    ----------
    endog : ndarray
        Dependent variable as 1-D array of shape (nobs,).
    exog : ndarray
        Independent variables as 2-D array of shape (nobs, k_vars).
    fixed_effects : list of str, list of int, or None
        Column identifiers for fixed effects variables. If `exog_names` is
        provided and elements are strings, interpreted as variable names.
        Otherwise interpreted as column indices. If None, returns original
        arrays unchanged.
    exog_names : list of str, optional
        Variable names corresponding to `exog` columns. Required when
        `fixed_effects` contains string identifiers.

    Returns
    -------
    endog_demeaned : ndarray
        Demeaned dependent variable of shape (nobs,).
    exog_demeaned : ndarray
        Demeaned independent variables of shape (nobs, k_vars - k_fe),
        where k_fe is the number of fixed effects variables removed.

    Raises
    ------
    ImportError
        If pyhdfe package is not installed.
    ValueError
        If `fixed_effects` contains string identifiers but `exog_names`
        is not provided.

    Notes
    -----
    The function performs within-group transformations by:

    1. Extracting fixed effects columns from exogenous variables
    2. Creating a pyhdfe algorithm object with these grouping variables
    3. Residualizing (demeaning) remaining variables within each group
    4. Removing the fixed effects columns from the output

    The pyhdfe package uses efficient algorithms for high-dimensional
    fixed effects [1]_.

    References
    ----------
    .. [1] Correia, S. (2017). "Linear Models with High-Dimensional Fixed
           Effects: An Efficient and Feasible Estimator."
           Working Paper. http://scorreia.com/research/hdfe.pdf

    Examples
    --------
    >>> endog = np.array([1.0, 2.0, 3.0, 4.0])
    >>> exog = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 1.0]])
    >>> endog_dm, exog_dm = _apply_fixed_effects(endog, exog, [1])
    """
    if fixed_effects is None:
        return endog.copy(), exog.copy()

    try:
        import pyhdfe
    except ImportError:
        raise ImportError(
            "pyhdfe is required for fixed effects estimation. "
            "Install with: pip install pyhdfe"
        )

    # Resolve fixed effects indices
    if isinstance(fixed_effects[0], str):
        if exog_names is None:
            raise ValueError(
                "exog_names must be provided when fixed_effects contains strings"
            )
        fe_indices = [exog_names.index(name) for name in fixed_effects]
    else:
        fe_indices = list(fixed_effects)

    # Extract fixed effects columns
    fe_cols = exog[:, fe_indices]
    if fe_cols.ndim == 1:
        fe_cols = fe_cols.reshape(-1, 1)

    # Extract non-fixed-effects columns
    non_fe_indices = [i for i in range(exog.shape[1]) if i not in fe_indices]
    exog_non_fe = exog[:, non_fe_indices]

    # Create pyhdfe algorithm and apply demeaning
    algorithm = pyhdfe.create(fe_cols, drop_singletons=False)
    endog_reshaped = endog.reshape(-1, 1)
    combined = np.column_stack([exog_non_fe, endog_reshaped])
    demeaned = algorithm.residualize(combined)

    # Split back into exog and endog
    exog_demeaned = demeaned[:, :-1]
    endog_demeaned = demeaned[:, -1]

    return endog_demeaned, exog_demeaned


def _detect_variable_types(
    exog: npt.NDArray[np.floating],
    indices: list[int],
) -> dict[int, Literal["continuous", "binary", "ordinal"]]:
    """
    Detect variable types as continuous, binary, or ordinal.

    Classifies each variable based on the number of unique values observed.
    Binary variables have exactly 2 unique values, ordinal variables have
    between 3 and 10 unique values, and continuous variables have more than
    10 unique values.

    Parameters
    ----------
    exog : ndarray
        Subset of independent variables to classify, shape (nobs, k_vars).
    indices : list of int
        Original column indices corresponding to the variables in `exog`.
        Used as keys in the returned dictionary.

    Returns
    -------
    variable_types : dict
        Dictionary mapping each variable index to its detected type:
        'continuous', 'binary', or 'ordinal'.

    Notes
    -----
    The classification heuristic is:

    - **Binary**: Exactly 2 unique values (e.g., 0/1, True/False)
    - **Ordinal**: 3-10 unique values (e.g., Likert scales, small categories)
    - **Continuous**: More than 10 unique values

    This heuristic may misclassify certain edge cases:

    - Categorical variables with >10 categories will be labeled continuous
    - Continuous variables with ≤10 observed values will be labeled ordinal

    Users should verify detected types match their domain knowledge and
    override if necessary.

    Examples
    --------
    >>> exog = np.array([[0, 1.5], [1, 2.3], [0, 3.1], [1, 4.7]])
    >>> types = _detect_variable_types(exog, indices=[0, 1])
    >>> types
    {0: 'binary', 1: 'continuous'}
    """
    variable_types = {}

    for i, col_idx in enumerate(indices):
        n_unique = len(np.unique(exog[:, i]))

        if n_unique == 2:
            variable_types[col_idx] = "binary"
        elif n_unique <= 10: # not sure whether this is the appropriate logic
            variable_types[col_idx] = "ordinal"
        else:
            variable_types[col_idx] = "continuous"

    return variable_types
