import numpy as np
import numpy.typing as npt
from typing import Literal

import pyhdfe


def _one_hot_encode_fe(fe_cols: np.ndarray) -> np.ndarray:
    """One-hot encode fixed effect columns, return (n_obs, total_levels) array."""
    dummies = []
    for col_idx in range(fe_cols.shape[1]):
        col = fe_cols[:, col_idx]
        unique_vals = np.unique(col)
        one_hot = np.zeros((len(col), len(unique_vals)), dtype=np.float64)
        for i, val in enumerate(unique_vals):
            one_hot[:, i] = (col == val).astype(np.float64)
        dummies.append(one_hot)
    return np.column_stack(dummies)


def _initialize_fixed_effects(fixed_effect_columns:npt.NDArray) -> pyhdfe.Algorithm:
    try:
        import pyhdfe
    except ImportError:
        raise ImportError(
            "pyhdfe is required for fixed effects estimation. "
            "Install with: pip install pyhdfe"
        )

    # Create pyhdfe algorithm and apply demeaning
    algorithm = pyhdfe.create(fixed_effect_columns, drop_singletons=False)
    return algorithm


def _apply_fixed_effects(
    endog: npt.NDArray[np.floating],
    exog: npt.NDArray[np.floating],
    algorithm,
    weights:npt.NDArray[np.floating] = None
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
    if algorithm is None:
        return endog, exog

    endog_reshaped = endog.reshape(-1, 1)
    combined = np.column_stack([exog, endog_reshaped])
    demeaned = algorithm.residualize(combined, weights=weights)

    # Split back into exog and endog
    exog_demeaned = demeaned[:, :-1]
    endog_demeaned = demeaned[:, -1]

    return endog_demeaned, exog_demeaned

def _redundant_columns(exog_demeaned: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
    """
    Identify redundant columns in demeaned exogenous variables.

    Redundant columns are those that are constant (zero variance)
    after demeaning, which can occur when fixed effects perfectly
    predict certain variables.

    Parameters
    ----------
    exog_demeaned : ndarray
        Demeaned independent variables of shape (nobs, k_vars).

    Returns
    -------
    redundant_indices : ndarray
        Indices of redundant columns in `exog_demeaned`.

    Examples
    --------
    >>> exog_dm = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    >>> redundant_idxs = _redundant_columns(exog_dm)
    >>> redundant_idxs
    array([0])
    """
    redundant_indices = np.where(np.all(np.abs(exog_demeaned) < 1e-6, axis=0))[0]

    return redundant_indices

def _delete_redundant(exog_demeaned, redundant_indices: npt.NDArray[np.integer]) -> npt.NDArray[np.floating]:
    """
    Remove redundant columns from demeaned exogenous variables.

    Identifies and removes columns that are constant (zero variance)
    after demeaning, which can occur when fixed effects perfectly
    predict certain variables.

    Parameters
    ----------
    exog_demeaned : ndarray
        Demeaned independent variables of shape (nobs, k_vars).

    Returns
    -------
    exog_cleaned : ndarray
        Exogenous variables with redundant columns removed,
        shape (nobs, k_cleaned), where k_cleaned ≤ k_vars.

    Examples
    --------
    >>> exog_dm = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    >>> exog_clean = _delete_redundant(exog_dm)
    >>> exog_clean
    array([[1.],
           [2.],
           [3.]])
    """
    exog_demeaned_clean = np.delete(exog_demeaned, redundant_indices, axis=1)
    return exog_demeaned_clean

def _adjust_indices_after_deletion(interest:list[int], redundant_indices: npt.NDArray[np.integer]) -> list[int]:
    """
    Adjust interest variable indices after removing redundant columns.

    Parameters
    ----------
    interest : list of int
        Original indices of interest variables.
    redundant_indices : ndarray
        Indices of columns removed from exogenous variables.

    Returns
    -------
    adjusted_interest : list of int
        Updated indices of interest variables after column removal.

    Examples
    --------
    >>> interest = [0, 2, 3]
    >>> redundant_indices = np.array([1])
    >>> adjusted = _adjust_indices_after_deletion(interest, redundant_indices)
    >>> adjusted
    [0, 1, 2]
    """
    redundant_set = set(redundant_indices.tolist())
    adjusted = []
    for idx in interest:
        if idx in redundant_set:
            raise ValueError(f"Interest variable at index {idx} was removed as redundant.")
        shift = np.sum(redundant_indices < idx)
        adjusted.append(idx - shift)
    return adjusted

def _adjust_names_after_deletion(names: list[str], redundant_indices: npt.NDArray[np.integer]) -> list[str]:
    """
    Adjust variable names list after removing redundant columns.

    Parameters
    ----------
    names : list of str
        Original variable names corresponding to exogenous variables.
    redundant_indices : ndarray
        Indices of columns removed from exogenous variables.

    Returns
    -------
    adjusted_names : list of str
        Updated variable names after column removal.

    Examples
    --------
    >>> names = ['x1', 'x2', 'x3']
    >>> redundant_indices = np.array([1])
    >>> adjusted_names = _adjust_names_after_deletion(names, redundant_indices)
    >>> adjusted_names
    ['x1', 'x3']
    """
    adjusted_names = [name for i, name in enumerate(names) if i not in redundant_indices]
    return adjusted_names

def _detect_variable_types(
    exog: npt.NDArray[np.floating],
    indices: list[int],
) -> dict[int, Literal["continuous", "binary"]]:
    """
    Detect whether variables are binary or continuous.

    Classifies each variable based on the number of unique values observed.
    Binary variables have exactly 2 unique values, all others are classified
    as continuous. Ordinal variables must be explicitly specified by the user
    and are not automatically detected.

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
        'binary' if exactly 2 unique values, 'continuous' otherwise.

    Notes
    -----
    The classification is simple and deterministic:

    - **Binary**: Exactly 2 unique values (e.g., 0/1, True/False)
    - **Continuous**: All other variables (including those with 3+ unique values)

    Ordinal variables are not automatically detected due to the difficulty in
    reliably distinguishing them from continuous or categorical variables.
    Users should explicitly specify ordinal variables through the model's
    `ordinal` parameter based on their domain knowledge.

    This approach avoids misclassification issues such as:
    - Continuous variables with limited observed variation being wrongly
      classified as ordinal
    - Ordered categorical variables with many levels (e.g., years of education)
      being wrongly classified as continuous

    Examples
    --------
    >>> exog = np.array([[0, 1.5], [1, 2.3], [0, 3.1], [1, 4.7]])
    >>> types = _detect_variable_types(exog, indices=[0, 1])
    >>> types
    {0: 'binary', 1: 'continuous'}
    
    >>> # Variable with 3 unique values is classified as continuous
    >>> exog = np.array([[1, 0], [2, 1], [3, 0], [2, 1]])
    >>> types = _detect_variable_types(exog, indices=[0, 1])
    >>> types
    {0: 'continuous', 1: 'binary'}
    """
    variable_types = {}

    for i, col_idx in enumerate(indices):
        n_unique = len(np.unique(exog[:, col_idx]))

        if n_unique == 2:
            variable_types[col_idx] = "binary"
        else:
            variable_types[col_idx] = "continuous"

    return variable_types
