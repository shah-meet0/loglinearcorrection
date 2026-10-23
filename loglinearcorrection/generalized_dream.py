"""Generalised GEAM and GPCAM estimation.

This module implements the continuous-treatment score in the ``Generalisation``
section of ``paper/2026-05-20-1401-draft-iv.md``.  It deliberately lives in a
separate module while the generalised estimator is experimental.

The special case ``phi=1``, ``psi=0`` and ``h(x)=1`` is the continuous DREAm
estimator already implemented in :mod:`loglinearcorrection.model`.  The GPCAM
implementation below follows the binary-treatment score in the same paper.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import KFold, StratifiedKFold

from .hypothesis import HypothesisTestResult, InfluenceFunctionDiffTest
from .model import DoublyRobustElasticityEstimatorModel
from .nonparametric import NNModelDensity, NNModelNuisance
from .utils import _apply_fixed_effects


Array = npt.NDArray[np.float64]
Policy = Callable[[Array], npt.ArrayLike]
Target = tuple[float, float]


@dataclass
class GeneralizedDREAMResults:
    """Results from :class:`GeneralizedDREAM`.

    The familiar DREAm attributes (``elasticities``, ``beta``,
    ``fold_results``, ``moments`` and ``standard_errors``) are retained.  The
    orthogonal-score influence functions used for inference are available as
    ``influence_functions``.
    """

    estimates: Array
    standard_errors: Array
    exog_names: list[str]
    endog_name: str
    interest_indices: list[int]
    variable_types: dict[int, str]
    beta: Array
    ols_results: Any
    phi: float
    psi: float
    policy_name: str
    n_folds: int
    nobs: int
    effective_nobs: float
    fold_results: list[dict[str, Any]]
    moments: Array
    influence_moments: Array
    influence_functions: Array
    covariance_matrix: Array
    fold_weights: Array
    k_weights: Array
    log_normalizer: float
    normalizer: float
    denominator: float
    _is_pandas: bool

    def __post_init__(self) -> None:
        z = float(stats.norm.ppf(0.975))
        names = [self.exog_names[i] for i in self.interest_indices]
        lower = self.estimates - z * self.standard_errors
        upper = self.estimates + z * self.standard_errors
        self.confidence_intervals = {
            name: (float(lower[j]), float(upper[j]))
            for j, name in enumerate(names)
        }
        self.fold_diagnostics = self.fold_results
        self.scores = self.moments

        if self._is_pandas:
            self.elasticities = pd.DataFrame(
                {
                    "estimate": self.estimates,
                    "std_err": self.standard_errors,
                    "type": [self.variable_types[i] for i in self.interest_indices],
                    "index": self.interest_indices,
                },
                index=pd.Index(names, name="variable"),
            )
        else:
            self.elasticities = self.estimates.copy()

    def get_elasticity(self, var_name: str) -> tuple[float, float]:
        """Return the estimate and standard error for ``var_name``."""
        names = [self.exog_names[i] for i in self.interest_indices]
        if var_name not in names:
            raise KeyError(f"Variable {var_name!r} not found")
        j = names.index(var_name)
        return float(self.estimates[j]), float(self.standard_errors[j])

    def summary(self) -> None:
        """Print a compact estimation summary."""
        print("=" * 78)
        print("Generalized DREAm (GEAM) Results")
        print(
            f"Dependent variable: {self.endog_name} | phi={self.phi:g} | "
            f"psi={self.psi:g} | policy={self.policy_name}"
        )
        print(
            f"Observations: {self.nobs} | effective observations: "
            f"{self.effective_nobs:.1f} | folds: {self.n_folds}"
        )
        print("-" * 78)
        print(f"{'Variable':<24} {'Estimate':>14} {'Std. Error':>14} {'95% CI':>23}")
        print("-" * 78)
        for j, idx in enumerate(self.interest_indices):
            lo, hi = self.confidence_intervals[self.exog_names[idx]]
            print(
                f"{self.exog_names[idx]:<24} {self.estimates[j]:>14.6g} "
                f"{self.standard_errors[j]:>14.6g} [{lo:>9.6g}, {hi:>9.6g}]"
            )
        print("=" * 78)

    def __repr__(self) -> str:
        return (
            "GeneralizedDREAMResults("
            f"phi={self.phi:g}, psi={self.psi:g}, "
            f"n_vars={len(self.interest_indices)}, nobs={self.nobs})"
        )


@dataclass
class JointGeneralizedDREAMResults:
    """Joint results for several ``(phi, psi)`` Generalized-DREAm targets.

    Targets are stacked in target-major, then variable-major order throughout
    ``covariance_matrix`` and ``influence_functions``.  Thus, with four targets
    and one variable of interest, the covariance matrix is 4 by 4; with ``k``
    variables it is ``(4*k)`` by ``(4*k)``.
    """

    targets: tuple[Target, ...]
    estimates: Array
    standard_errors: Array
    exog_names: list[str]
    endog_name: str
    interest_indices: list[int]
    variable_types: dict[int, str]
    beta: Array
    ols_results: Any
    policy_name: str
    n_folds: int
    nobs: int
    effective_nobs: float
    fold_results: list[dict[str, Any]]
    moments: Array
    influence_moments: Array
    influence_functions: Array
    covariance_matrix: Array
    fold_weights: Array
    k_weights: Array
    log_normalizers: Array
    normalizers: Array
    denominators: Array
    _is_pandas: bool

    def __post_init__(self) -> None:
        n_targets = len(self.targets)
        n_interest = len(self.interest_indices)
        expected = (n_targets, n_interest)
        if self.estimates.shape != expected or self.standard_errors.shape != expected:
            raise ValueError(
                "estimates and standard_errors must have shape "
                f"{expected}; got {self.estimates.shape} and {self.standard_errors.shape}"
            )
        expected_covariance = (n_targets * n_interest, n_targets * n_interest)
        if self.covariance_matrix.shape != expected_covariance:
            raise ValueError(
                f"covariance_matrix must have shape {expected_covariance}; "
                f"got {self.covariance_matrix.shape}"
            )

        names = self.interest_names
        z_value = float(stats.norm.ppf(0.975))
        records: list[dict[str, Any]] = []
        confidence_intervals: dict[tuple[float, float, str], tuple[float, float]] = {}
        for target_index, (phi, psi) in enumerate(self.targets):
            for variable_index, name in enumerate(names):
                estimate = float(self.estimates[target_index, variable_index])
                std_error = float(self.standard_errors[target_index, variable_index])
                lower = estimate - z_value * std_error
                upper = estimate + z_value * std_error
                confidence_intervals[(phi, psi, name)] = (lower, upper)
                records.append(
                    {
                        "phi": phi,
                        "psi": psi,
                        "variable": name,
                        "estimate": estimate,
                        "std_error": std_error,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "type": self.variable_types[self.interest_indices[variable_index]],
                        "variable_index": self.interest_indices[variable_index],
                    }
                )

        self.confidence_intervals = confidence_intervals
        self.elasticities = pd.DataFrame.from_records(records).set_index(
            ["phi", "psi", "variable"]
        )
        self.fold_diagnostics = self.fold_results
        self.scores = self.moments

    @property
    def interest_names(self) -> list[str]:
        return [self.exog_names[index] for index in self.interest_indices]

    @property
    def parameter_labels(self) -> list[tuple[float, float, str]]:
        return [
            (phi, psi, name)
            for phi, psi in self.targets
            for name in self.interest_names
        ]

    def _target_position(self, phi: float, psi: float) -> int:
        target = (float(phi), float(psi))
        try:
            return self.targets.index(target)
        except ValueError as exc:
            raise KeyError(f"Target (phi={phi:g}, psi={psi:g}) was not estimated") from exc

    def _variable_position(self, var_name: str) -> int:
        try:
            return self.interest_names.index(var_name)
        except ValueError as exc:
            raise KeyError(f"Variable {var_name!r} not found") from exc

    def parameter_position(self, phi: float, psi: float, var_name: str) -> int:
        target_position = self._target_position(phi, psi)
        variable_position = self._variable_position(var_name)
        return target_position * len(self.interest_indices) + variable_position

    def get_elasticity(
        self, var_name: str, *, phi: float, psi: float
    ) -> tuple[float, float]:
        """Return a target estimate and its standard error."""
        target_position = self._target_position(phi, psi)
        variable_position = self._variable_position(var_name)
        return (
            float(self.estimates[target_position, variable_position]),
            float(self.standard_errors[target_position, variable_position]),
        )

    def difference(
        self,
        var_name: str,
        target_a: Target,
        target_b: Target,
    ) -> dict[str, float]:
        """Return a covariance-aware two-sided test of two target estimates."""
        a = self.parameter_position(*target_a, var_name)
        b = self.parameter_position(*target_b, var_name)
        flat_estimates = self.estimates.reshape(-1)
        difference = float(flat_estimates[a] - flat_estimates[b])
        variance = float(
            self.covariance_matrix[a, a]
            + self.covariance_matrix[b, b]
            - 2.0 * self.covariance_matrix[a, b]
        )
        std_error = float(np.sqrt(max(variance, 0.0)))
        z_stat = difference / std_error if std_error > 0.0 else np.nan
        p_value = float(2.0 * stats.norm.sf(abs(z_stat))) if np.isfinite(z_stat) else np.nan
        return {
            "difference": difference,
            "std_error": std_error,
            "z_stat": float(z_stat),
            "p_value": p_value,
        }

    def difference_test(
        self,
        var_name: str,
        *,
        phi: float,
        psi: float,
        comparator_estimate: float,
        comparator_influence_functions: npt.ArrayLike,
        comparator_name: str = "comparator",
    ) -> HypothesisTestResult:
        """Test one GEAM target against a same-sample comparator.

        The comparator influence function must be on the coefficient scale of
        ``comparator_estimate``.  Continuous GEAM targets are already on that
        scale, so unlike GPCAM comparisons no exponential transformation is
        applied to OLS or PPML coefficients.
        """
        parameter_position = self.parameter_position(phi, psi, var_name)
        target_position = self._target_position(phi, psi)
        variable_position = self._variable_position(var_name)
        comparator_influence = np.asarray(
            comparator_influence_functions, dtype=np.float64
        ).reshape(-1)
        if comparator_influence.shape != (self.nobs,):
            raise ValueError(
                "comparator_influence_functions must have length "
                f"{self.nobs}; got {comparator_influence.shape}"
            )
        test = InfluenceFunctionDiffTest(
            n=self.nobs,
            k=1,
            coefs=np.asarray(
                [
                    self.estimates[target_position, variable_position],
                    comparator_estimate,
                ]
            ),
            influence_functions=np.column_stack(
                [
                    self.influence_functions[:, parameter_position],
                    comparator_influence,
                ]
            ),
            weights=self.fold_weights,
        )
        result = test.results[0]
        result.description = (
            f"Wald test: GEAM(phi={phi:g}, psi={psi:g}) = {comparator_name}"
        )
        return result

    def ols_difference_test(
        self,
        var_name: str,
        *,
        phi: float,
        psi: float,
    ) -> HypothesisTestResult:
        """Test a GEAM target against the corresponding log-OLS coefficient."""
        variable_position = self._variable_position(var_name)
        coefficient_position = self.interest_indices[variable_position]
        design = np.asarray(self.ols_results.model.exog, dtype=np.float64)
        residual = np.asarray(self.ols_results.resid, dtype=np.float64).reshape(-1)
        if design.shape[0] != self.nobs or residual.shape != (self.nobs,):
            raise ValueError("OLS result does not align with the GEAM sample")

        probability_weight = self.fold_weights / self.fold_weights.sum()
        bread = (design.T * probability_weight) @ design
        selector = np.zeros(design.shape[1], dtype=np.float64)
        selector[coefficient_position] = 1.0
        bread_column = np.linalg.pinv(bread, rcond=1e-10) @ selector
        coefficient_influence = (design @ bread_column) * residual
        coefficient = float(self.ols_results.params[coefficient_position])
        return self.difference_test(
            var_name,
            phi=phi,
            psi=psi,
            comparator_estimate=coefficient,
            comparator_influence_functions=coefficient_influence,
            comparator_name="OLS coefficient",
        )

    def estimates_frame(self) -> pd.DataFrame:
        """Return one row per target and variable, including covariance rows."""
        frame = self.elasticities.reset_index().copy()
        labels = self.parameter_labels
        covariance = self.covariance_matrix
        for column_position, (phi, psi, name) in enumerate(labels):
            label = f"cov_phi_{phi:g}_psi_{psi:g}_{name}"
            frame[label] = covariance[:, column_position]
        return frame

    def covariance_frame(self) -> pd.DataFrame:
        """Return the full joint covariance matrix in tidy long form."""
        records: list[dict[str, Any]] = []
        labels = self.parameter_labels
        for row, column in combinations(range(len(labels)), 2):
            left = labels[row]
            right = labels[column]
            records.append(
                {
                    "phi_a": left[0],
                    "psi_a": left[1],
                    "variable_a": left[2],
                    "phi_b": right[0],
                    "psi_b": right[1],
                    "variable_b": right[2],
                    "covariance": float(self.covariance_matrix[row, column]),
                }
            )
        for position, label in enumerate(labels):
            records.append(
                {
                    "phi_a": label[0],
                    "psi_a": label[1],
                    "variable_a": label[2],
                    "phi_b": label[0],
                    "psi_b": label[1],
                    "variable_b": label[2],
                    "covariance": float(self.covariance_matrix[position, position]),
                }
            )
        return pd.DataFrame.from_records(records)

    def summary(self) -> None:
        """Print a compact joint-estimation summary."""
        print("=" * 88)
        print("Joint Generalized DREAm (GEAM) Results")
        print(
            f"Dependent variable: {self.endog_name} | targets: {self.targets} | "
            f"policy={self.policy_name}"
        )
        print(
            f"Observations: {self.nobs} | effective observations: "
            f"{self.effective_nobs:.1f} | folds: {self.n_folds}"
        )
        print("-" * 88)
        print(self.elasticities[["estimate", "std_error", "ci_lower", "ci_upper"]].to_string())
        print("=" * 88)

    def __repr__(self) -> str:
        return (
            "JointGeneralizedDREAMResults("
            f"targets={self.targets}, n_vars={len(self.interest_indices)}, "
            f"nobs={self.nobs})"
        )


@dataclass
class GPCAMResults:
    """Results from :class:`GeneralizedPCAM`."""

    estimate: float
    standard_error: float
    phi: float
    psi: float
    treatment_name: str
    treatment_index: int
    exog_names: list[str]
    endog_name: str
    beta: Array
    ols_results: Any
    n_folds: int
    nobs: int
    effective_nobs: float
    fold_results: list[dict[str, Any]]
    moments: Array
    influence_functions: Array
    covariance_matrix: Array
    fold_weights: Array
    arm_means: Array
    log_arm_means: Array
    score_derivative: float
    score_scale_log: float

    def __post_init__(self) -> None:
        z_value = float(stats.norm.ppf(0.975))
        lower = self.estimate - z_value * self.standard_error
        upper = self.estimate + z_value * self.standard_error
        self.confidence_interval = (float(lower), float(upper))
        self.confidence_intervals = {
            self.treatment_name: self.confidence_interval
        }
        self.percentage_change = self.estimate
        self.estimates = np.asarray([self.estimate], dtype=np.float64)
        self.standard_errors = np.asarray([self.standard_error], dtype=np.float64)
        self.scores = self.moments
        self.fold_diagnostics = self.fold_results
        self.treatment_beta = float(self.beta[self.treatment_index])

    def get_percentage_change(self) -> tuple[float, float]:
        """Return the percentage-change estimate and its standard error."""
        return self.estimate, self.standard_error

    def difference_test(
        self,
        comparator_estimate: float,
        comparator_influence_functions: npt.ArrayLike,
        *,
        comparator_name: str = "comparator",
    ) -> HypothesisTestResult:
        """Test equality with another same-sample percentage-change estimate.

        ``comparator_influence_functions`` must be on the scale of
        ``comparator_estimate``.  In particular, for a log-link coefficient
        ``beta`` compared as ``exp(beta) - 1``, its coefficient influence
        function must also be multiplied by ``exp(beta)``.  The test retains
        the empirical covariance between GPCAM and the comparator.
        """
        comparator_influence = np.asarray(
            comparator_influence_functions, dtype=np.float64
        ).reshape(-1)
        if comparator_influence.shape != (self.nobs,):
            raise ValueError(
                "comparator_influence_functions must have length "
                f"{self.nobs}; got {comparator_influence.shape}"
            )
        test = InfluenceFunctionDiffTest(
            n=self.nobs,
            k=1,
            coefs=np.asarray([self.estimate, comparator_estimate]),
            influence_functions=np.column_stack(
                [self.influence_functions, comparator_influence]
            ),
            weights=self.fold_weights,
        )
        result = test.results[0]
        result.description = (
            f"Wald test: GPCAM(phi={self.phi:g}, psi={self.psi:g}) "
            f"= {comparator_name}"
        )
        return result

    def ols_difference_test(self) -> HypothesisTestResult:
        """Test GPCAM against ``exp(beta_OLS) - 1`` on the same sample."""
        design = np.asarray(self.ols_results.model.exog, dtype=np.float64)
        residual = np.asarray(self.ols_results.resid, dtype=np.float64).reshape(-1)
        if design.shape[0] != self.nobs or residual.shape != (self.nobs,):
            raise ValueError("OLS result does not align with the GPCAM sample")

        probability_weight = self.fold_weights / self.fold_weights.sum()
        bread = (design.T * probability_weight) @ design
        treatment_selector = np.zeros(design.shape[1], dtype=np.float64)
        treatment_selector[self.treatment_index] = 1.0
        bread_column = np.linalg.pinv(bread) @ treatment_selector
        coefficient_influence = (design @ bread_column) * residual
        percentage_estimate = float(np.expm1(self.treatment_beta))
        percentage_influence = np.exp(self.treatment_beta) * coefficient_influence
        return self.difference_test(
            percentage_estimate,
            percentage_influence,
            comparator_name="exp(OLS coefficient) - 1",
        )

    def summary(self) -> None:
        """Print a compact estimation summary."""
        lower, upper = self.confidence_interval
        print("=" * 78)
        print("Generalized Percentage Change of Aggregate Means (GPCAM) Results")
        print(
            f"Dependent variable: {self.endog_name} | treatment: "
            f"{self.treatment_name} | phi={self.phi:g} | psi={self.psi:g}"
        )
        print(
            f"Observations: {self.nobs} | effective observations: "
            f"{self.effective_nobs:.1f} | folds: {self.n_folds}"
        )
        print("-" * 78)
        print(
            f"Percentage change: {self.estimate:.6g} "
            f"(SE {self.standard_error:.6g}; 95% CI "
            f"[{lower:.6g}, {upper:.6g}])"
        )
        print("=" * 78)

    def __repr__(self) -> str:
        return (
            "GPCAMResults("
            f"phi={self.phi:g}, psi={self.psi:g}, "
            f"estimate={self.estimate:.6g}, nobs={self.nobs})"
        )


class GeneralizedDREAM(DoublyRobustElasticityEstimatorModel):
    r"""Generalised DREAm for continuous variables of interest.

    The estimator targets

    .. math::

        \theta_{\phi,\psi}(h)
        = \frac{E[h(X) M_\phi(X)^\psi\epsilon_\phi(X)]}
               {E[M_\phi(X)^\psi]}.

    Parameters
    ----------
    endog, exog, weights, fixed_effects, interest
        As in :class:`~loglinearcorrection.model.DoublyRobustElasticityEstimatorModel`.
        Every variable in ``interest`` must be continuous and non-constant.
    phi : float, default 1
        Inner power-mean parameter.  ``phi=1`` is the arithmetic mean and
        ``phi=0`` is the geometric-mean limit.
    psi : float, default 0
        Outer power-mean parameter.  ``psi=0`` gives equal weighting over the
        regressor distribution.
    h : callable, optional
        Policy function.  It receives the two-dimensional, non-fixed-effect
        design matrix and may return a scalar, ``(n,)``, ``(n, 1)`` or
        ``(n, n_interest)`` array.  The default is ``h(x)=1``.
    h_derivative : callable, optional
        Own-coordinate derivative of ``h`` with the same output conventions.
        If omitted for a non-constant policy, a central numerical derivative is
        used.

    Notes
    -----
    Derivatives are taken with respect to the columns supplied in ``exog``.  To
    estimate elasticities rather than semi-elasticities, the variables of
    interest should therefore be supplied as log regressors, as in the paper.

    For ``phi != 0`` the nuisance regression estimates
    ``m_phi(x)=E[exp(phi*u)|x]``.  At ``phi=0`` the implementation uses the
    continuous limit, estimating ``E[u|x]`` and using ``u-E[u|x]`` as ``R_0``.
    """

    def __init__(
        self,
        endog: npt.ArrayLike,
        exog: npt.ArrayLike,
        weights: npt.ArrayLike | None = None,
        fixed_effects: list[str] | list[int] | None = None,
        interest: list[str] | list[int] | str | int | None = None,
        *,
        phi: float = 1.0,
        psi: float = 0.0,
        h: Policy | None = None,
        h_derivative: Policy | None = None,
        **kwargs: Any,
    ) -> None:
        self.phi = self._finite_scalar(phi, "phi")
        self.psi = self._finite_scalar(psi, "psi")
        self.h = h
        self.h_derivative = h_derivative
        self.policy_name = "1" if h is None else getattr(h, "__name__", type(h).__name__)

        y = np.asarray(endog, dtype=np.float64).ravel()
        if y.size == 0 or not np.all(np.isfinite(y)) or np.any(y <= 0):
            raise ValueError("endog must contain finite, strictly positive values")

        # model.py currently evaluates a numpy weight vector in boolean context.
        # Initialise the shared data structure without weights, then install and
        # apply validated weights here without changing the existing module.
        supplied_weights = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
        super().__init__(
            endog=endog,
            exog=exog,
            weights=None,
            fixed_effects=fixed_effects,
            interest=interest,
            ordinal=None,
            **kwargs,
        )

        if not np.all(np.isfinite(self.exog)):
            raise ValueError("exog must contain only finite values")

        if supplied_weights is not None:
            self._install_weights(supplied_weights)

        interest_indices = list(self.interest)
        if not interest_indices:
            raise ValueError("At least one variable of interest is required")

        unsupported = [
            self.exog_names[j]
            for j in interest_indices
            if self.variable_types.get(j) != "continuous"
            or np.unique(self.exog[:, j]).size < 3
        ]
        if unsupported:
            names = ", ".join(unsupported)
            raise NotImplementedError(
                "GeneralizedDREAM currently supports continuous, non-constant "
                f"variables of interest only; unsupported: {names}"
            )

    @staticmethod
    def _finite_scalar(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _install_weights(self, weights: Array) -> None:
        if weights.shape != (self.nobs,):
            raise ValueError(f"weights must have length {self.nobs}")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("weights must be finite, non-negative, and have positive sum")

        self.weights = weights
        if self.fixed_effects is None:
            self.ols_res = sm.WLS(np.log(self.endog), self.exog, weights=weights).fit()
        else:
            y_dm, x_dm = _apply_fixed_effects(
                np.log(self.endog),
                self.exog,
                algorithm=self.fixed_effects,
                weights=weights.reshape(-1, 1),
            )
            self.temp_exog_demeaned = x_dm
            self.ols_res = sm.WLS(y_dm, x_dm, weights=weights).fit()
        self.beta = np.asarray(self.ols_res.params, dtype=np.float64)

    def _prepare_m_params(
        self,
        m_params: Optional[Mapping[str, Any]],
        *,
        phi: float | None = None,
    ) -> dict[str, Any]:
        target_phi = self.phi if phi is None else self._finite_scalar(phi, "phi")
        supplied = deepcopy(dict(m_params)) if m_params is not None else None
        parsed = self._parse_m_params(supplied)
        user_set_activation = bool(
            m_params is not None
            and isinstance(m_params.get("arch_params"), Mapping)
            and "output_activation" in m_params["arch_params"]
        )
        if not user_set_activation:
            parsed["arch_params"]["output_activation"] = "identity"
        if target_phi != 0.0:
            parsed["arch_params"].setdefault("positive_log_link", True)
            parsed["arch_params"].setdefault("log_link_max_abs", 3.0)
        return parsed

    def _prepare_density_params(
        self, density_params: Optional[Mapping[str, Any]]
    ) -> tuple[type, dict[str, Any]]:
        supplied = deepcopy(dict(density_params)) if density_params is not None else {}
        model_class = supplied.pop("model_class", NNModelDensity)
        parsed = self._parse_density_params(supplied or None)
        return model_class, parsed

    @staticmethod
    def _coerce_derivative_output(
        prediction: npt.ArrayLike,
        derivative: npt.ArrayLike,
        n: int,
        k: int,
    ) -> tuple[Array, Array]:
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        derivative = np.asarray(derivative, dtype=np.float64)
        if prediction.shape != (n,):
            raise ValueError(f"Nuisance predict/derivative returned {prediction.shape}; expected ({n},)")
        if derivative.ndim == 1 and k == 1:
            derivative = derivative.reshape(-1, 1)
        if derivative.shape != (n, k):
            raise ValueError(
                f"Nuisance derivative returned {derivative.shape}; expected ({n}, {k})"
            )
        return prediction, derivative

    @staticmethod
    def _coerce_policy_output(value: npt.ArrayLike, n: int, k: int, name: str) -> Array:
        value = np.asarray(value, dtype=np.float64)
        if value.ndim == 0:
            result = np.full((n, k), float(value))
        elif value.shape == (n,):
            result = np.repeat(value[:, None], k, axis=1)
        elif value.shape == (k,) and n != k:
            result = np.repeat(value[None, :], n, axis=0)
        elif value.shape == (n, 1):
            result = np.repeat(value, k, axis=1)
        elif value.shape == (n, k):
            result = value
        else:
            raise ValueError(
                f"{name} returned shape {value.shape}; expected a scalar, ({n},), "
                f"({n}, 1), or ({n}, {k})"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} returned non-finite values")
        return result

    def _evaluate_policy(self, x: Array, interest_indices: list[int]) -> tuple[Array, Array]:
        n, k = x.shape[0], len(interest_indices)
        if self.h is None:
            return np.ones((n, k)), np.zeros((n, k))

        h = self._coerce_policy_output(self.h(x), n, k, "h")
        if self.h_derivative is not None:
            h_prime = self._coerce_policy_output(
                self.h_derivative(x), n, k, "h_derivative"
            )
            return h, h_prime

        h_prime = np.empty((n, k), dtype=np.float64)
        eps = np.cbrt(np.finfo(np.float64).eps)
        for out_j, var_j in enumerate(interest_indices):
            step = eps * np.maximum(1.0, np.abs(x[:, var_j]))
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[:, var_j] += step
            x_minus[:, var_j] -= step
            h_plus = self._coerce_policy_output(self.h(x_plus), n, k, "h")
            h_minus = self._coerce_policy_output(self.h(x_minus), n, k, "h")
            h_prime[:, out_j] = (h_plus[:, out_j] - h_minus[:, out_j]) / (2.0 * step)
        return h, h_prime

    def _make_splits(
        self, n_folds: int, random_state: Optional[int]
    ) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        if not isinstance(n_folds, int) or n_folds < 2 or n_folds > self.nobs:
            raise ValueError("n_folds must be an integer between 2 and nobs")

        if self.fe_labels is not None:
            _, counts = np.unique(self.fe_labels, return_counts=True)
            if np.all(counts >= n_folds):
                splitter = StratifiedKFold(
                    n_splits=n_folds, shuffle=True, random_state=random_state
                )
                iterator = splitter.split(self.exog, self.fe_labels)
            else:
                splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                iterator = splitter.split(self.exog)
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            iterator = splitter.split(self.exog)
        return [(np.asarray(train), np.asarray(test)) for train, test in iterator]

    def _fold_ols_residuals(
        self,
        train_idx: npt.NDArray[np.int64],
        test_idx: npt.NDArray[np.int64],
        weight_fold: Array,
    ) -> tuple[Any, Array, Array, Array, Array, Array]:
        """Fold OLS and log residuals net of fixed effects.

        With fixed effects, log(y) and X are demeaned on the full sample using
        ``weight_fold`` (held-out observations carry weight 1e-10), as in
        :mod:`loglinearcorrection.model`, so the fixed-effect means come from
        the training fold.  The residuals are then the demeaned residuals, and
        ``fe_train``/``fe_test`` hold each observation's estimated
        fixed-effect component, ``(log y - x beta) - residual``.  Without fixed
        effects the components are zero and the residuals are
        ``log y - x beta``.
        """
        log_y = np.log(self.endog)
        train_weights = weight_fold[train_idx] if self.weights is not None else None
        if self.fixed_effects is None:
            x_ols, log_y_ols = self.exog, log_y
        else:
            log_y_ols, x_ols = _apply_fixed_effects(
                log_y,
                self.exog,
                algorithm=self.fixed_effects,
                weights=weight_fold.reshape(-1, 1),
            )
            log_y_ols = np.asarray(log_y_ols, dtype=np.float64).reshape(-1)
            x_ols = np.asarray(x_ols, dtype=np.float64)

        ols = (
            sm.WLS(log_y_ols[train_idx], x_ols[train_idx], weights=train_weights)
            if train_weights is not None
            else sm.OLS(log_y_ols[train_idx], x_ols[train_idx])
        ).fit()
        beta = np.asarray(ols.params, dtype=np.float64)

        residual = log_y_ols - x_ols @ beta
        fe_component = (log_y - self.exog @ beta) - residual
        return (
            ols,
            beta,
            residual[train_idx],
            residual[test_idx],
            fe_component[train_idx],
            fe_component[test_idx],
        )

    def _process_generalized_fold_common(
        self,
        train_idx: npt.NDArray[np.int64],
        test_idx: npt.NDArray[np.int64],
        weight_fold: Array,
        interest_indices: list[int],
        density_model_class: type,
        density_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Fit fold components shared by every ``(phi, psi)`` target."""
        ols, beta, residual_train, residual_test, _, fe_test = self._fold_ols_residuals(
            train_idx, test_idx, weight_fold
        )

        x_test = self.exog[test_idx]
        x_aug_train = self.exog_augmented[train_idx]
        x_aug_test = self.exog_augmented[test_idx]

        density_model = density_model_class(
            variable_types=self.variable_types, **density_params["arch_params"]
        )
        density_results = density_model.fit(
            x_aug_train,
            interest=interest_indices,
            **density_params["fit_params"],
        )
        density_score = np.asarray(
            density_results.alpha_weight(x_aug_test, interest_indices), dtype=np.float64
        )
        if density_score.ndim == 1 and len(interest_indices) == 1:
            density_score = density_score.reshape(-1, 1)
        expected_shape = (len(test_idx), len(interest_indices))
        if density_score.shape != expected_shape:
            raise ValueError(
                f"Density score returned {density_score.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(density_score)):
            raise ValueError("The fitted density score returned non-finite values")

        h, h_prime = self._evaluate_policy(x_test, interest_indices)
        return {
            "beta": beta,
            "ols_results": ols,
            "f_results": density_results,
            "density_score": density_score,
            "h": h,
            "h_prime": h_prime,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "_x_aug_train": x_aug_train,
            "_x_test": x_test,
            "_residual_train": residual_train,
            "_residual_test": residual_test,
            "_fe_test": fe_test,
        }

    def _process_phi_fold(
        self,
        common: Mapping[str, Any],
        interest_indices: list[int],
        phi: float,
        m_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Fit the nuisance component shared by all targets with one ``phi``."""
        residual_train = np.asarray(common["_residual_train"], dtype=np.float64)
        residual_test = np.asarray(common["_residual_test"], dtype=np.float64)
        x_aug_train = np.asarray(common["_x_aug_train"], dtype=np.float64)
        x_test = np.asarray(common["_x_test"], dtype=np.float64)
        test_idx = np.asarray(common["test_idx"])
        beta = np.asarray(common["beta"], dtype=np.float64)

        if phi == 0.0:
            nuisance_target = residual_train
            transformed_test = residual_test
        else:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                nuisance_target = np.exp(phi * residual_train)
                transformed_test = np.exp(phi * residual_test)
            if not np.all(np.isfinite(nuisance_target)) or not np.all(
                np.isfinite(transformed_test)
            ):
                raise FloatingPointError(
                    "exp(phi * residual) overflowed; rescale the outcome or use a less "
                    "extreme phi"
                )

        m_model_class = m_params.get("model_class", NNModelNuisance)
        m_model = m_model_class(
            variable_types=self.variable_types, **m_params["arch_params"]
        )
        m_results = m_model.fit(
            x_aug_train, nuisance_target, **m_params["fit_params"]
        )
        m_hat, m_prime = m_results.derivative(
            self.exog_augmented[test_idx], interest_indices
        )
        m_hat, m_prime = self._coerce_derivative_output(
            m_hat, m_prime, len(test_idx), len(interest_indices)
        )
        if not np.all(np.isfinite(m_hat)) or not np.all(np.isfinite(m_prime)):
            raise ValueError("The fitted nuisance function returned non-finite values")

        # Fixed-effect component added back so M_phi(x) is the power mean of y itself
        linear_predictor = x_test @ beta + np.asarray(common["_fe_test"], dtype=np.float64)
        beta_interest = beta[interest_indices]
        if phi == 0.0:
            conditional_elasticity = beta_interest[None, :] + m_prime
            log_inner_mean = linear_predictor + m_hat
            r_phi = transformed_test - m_hat
        else:
            if np.any(m_hat <= 0):
                raise ValueError(
                    "m_phi must be strictly positive. Use a positive-output "
                    "nuisance model (the default neural nuisance uses a bounded "
                    "log link here)."
                )
            conditional_elasticity = (
                beta_interest[None, :] + m_prime / (phi * m_hat[:, None])
            )
            log_inner_mean = linear_predictor + np.log(m_hat) / phi
            r_phi = (transformed_test - m_hat) / (phi * m_hat)

        return {
            "phi": phi,
            "m_results": m_results,
            "conditional_elasticity": conditional_elasticity,
            "log_inner_mean": log_inner_mean,
            "r_phi": r_phi,
        }

    @staticmethod
    def _public_common_fold(common: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in common.items() if not key.startswith("_")}

    def _process_generalized_fold(
        self,
        train_idx: npt.NDArray[np.int64],
        test_idx: npt.NDArray[np.int64],
        weight_fold: Array,
        interest_indices: list[int],
        m_params: dict[str, Any],
        density_model_class: type,
        density_params: dict[str, Any],
    ) -> dict[str, Any]:
        common = self._process_generalized_fold_common(
            train_idx,
            test_idx,
            weight_fold,
            interest_indices,
            density_model_class,
            density_params,
        )
        phi_result = self._process_phi_fold(
            common, interest_indices, self.phi, m_params
        )
        return {**self._public_common_fold(common), **phi_result}

    @staticmethod
    def _solve_target(
        *,
        conditional_elasticity: Array,
        density_score: Array,
        h: Array,
        h_prime: Array,
        log_inner_mean: Array,
        r_phi: Array,
        fold_weights: Array,
        psi: float,
    ) -> dict[str, Any]:
        positive_weight = fold_weights > 0
        if not np.any(positive_weight):
            raise ValueError("At least one observation must have positive weight")
        log_power_weight = psi * log_inner_mean
        if not np.all(np.isfinite(log_power_weight[positive_weight])):
            raise FloatingPointError("The estimated power-mean weights are non-finite")

        shift = float(np.max(log_power_weight[positive_weight]))
        scaled_weight = np.zeros(len(fold_weights), dtype=np.float64)
        scaled_weight[positive_weight] = np.exp(
            log_power_weight[positive_weight] - shift
        )
        scaled_normalizer = float(np.average(scaled_weight, weights=fold_weights))
        if not np.isfinite(scaled_normalizer) or scaled_normalizer <= 0:
            raise FloatingPointError("Could not normalize the estimated power-mean weights")
        k_weight = scaled_weight / scaled_normalizer
        log_normalizer = shift + float(np.log(scaled_normalizer))
        normalizer = (
            float(np.exp(log_normalizer))
            if log_normalizer <= np.log(np.finfo(np.float64).max)
            else float("inf")
        )

        a_term = (
            k_weight[:, None] * h * conditional_elasticity
            - k_weight[:, None]
            * (h_prime + h * density_score)
            * r_phi[:, None]
        )
        b_term = 1.0 + psi * k_weight * r_phi
        if not np.all(np.isfinite(a_term)) or not np.all(np.isfinite(b_term)):
            raise FloatingPointError("The generalized DREAm score contains non-finite values")

        denominator = float(np.average(b_term, weights=fold_weights))
        if not np.isfinite(denominator) or abs(denominator) <= np.sqrt(
            np.finfo(float).eps
        ):
            raise FloatingPointError(
                "The sample derivative of the generalized score is numerically zero"
            )
        estimates = np.average(a_term, axis=0, weights=fold_weights) / denominator
        moments = a_term - b_term[:, None] * estimates[None, :]
        normalization_adjustment = (k_weight - 1.0)[:, None] * estimates[None, :]
        influence_moments = moments - normalization_adjustment
        influence = influence_moments / denominator

        probability_weight = fold_weights / fold_weights.sum()
        influence -= np.sum(
            probability_weight[:, None] * influence, axis=0, keepdims=True
        )
        return {
            "estimates": estimates,
            "moments": moments,
            "influence_moments": influence_moments,
            "influence_functions": influence,
            "k_weights": k_weight,
            "log_normalizer": log_normalizer,
            "normalizer": normalizer,
            "denominator": denominator,
        }

    def fit(
        self,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        m_params: Optional[Mapping[str, Any]] = None,
        density_params: Optional[Mapping[str, Any]] = None,
    ) -> GeneralizedDREAMResults:
        """Fit the cross-fitted continuous GEAM estimator.

        ``m_params`` follows the existing DREAm structure and additionally may
        contain ``model_class``.  ``density_params`` follows the existing
        density configuration and may also contain ``model_class``, which is
        useful for alternative score estimators and isolated testing.
        """
        interest_indices = list(self.interest)
        k_interest = len(interest_indices)
        m_config = self._prepare_m_params(m_params)
        density_model_class, density_config = self._prepare_density_params(density_params)
        splits = self._make_splits(n_folds, random_state)

        fold_results: list[dict[str, Any]] = []
        for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
            weight_fold = np.ones(self.nobs, dtype=np.float64)
            weight_fold[test_idx] = 1e-10
            if self.weights is not None:
                weight_fold *= self.weights
            print(f"Processing generalized DREAm fold {fold_number}/{n_folds}")
            fold_results.append(
                self._process_generalized_fold(
                    train_idx,
                    test_idx,
                    weight_fold,
                    interest_indices,
                    m_config,
                    density_model_class,
                    density_config,
                )
            )

        conditional_elasticity = np.empty((self.nobs, k_interest), dtype=np.float64)
        density_score = np.empty_like(conditional_elasticity)
        h = np.empty_like(conditional_elasticity)
        h_prime = np.empty_like(conditional_elasticity)
        log_inner_mean = np.empty(self.nobs, dtype=np.float64)
        r_phi = np.empty(self.nobs, dtype=np.float64)

        for fold in fold_results:
            idx = fold["test_idx"]
            conditional_elasticity[idx] = fold["conditional_elasticity"]
            density_score[idx] = fold["density_score"]
            h[idx] = fold["h"]
            h_prime[idx] = fold["h_prime"]
            log_inner_mean[idx] = fold["log_inner_mean"]
            r_phi[idx] = fold["r_phi"]

        fold_weights = (
            np.ones(self.nobs, dtype=np.float64)
            if self.weights is None
            else np.asarray(self.weights, dtype=np.float64).copy()
        )
        target_solution = self._solve_target(
            conditional_elasticity=conditional_elasticity,
            density_score=density_score,
            h=h,
            h_prime=h_prime,
            log_inner_mean=log_inner_mean,
            r_phi=r_phi,
            fold_weights=fold_weights,
            psi=self.psi,
        )
        estimates = target_solution["estimates"]
        moments = target_solution["moments"]
        influence_moments = target_solution["influence_moments"]
        influence = target_solution["influence_functions"]
        k_weight = target_solution["k_weights"]
        log_normalizer = target_solution["log_normalizer"]
        normalizer = target_solution["normalizer"]
        denominator = target_solution["denominator"]
        probability_weight = fold_weights / fold_weights.sum()
        weighted_influence = probability_weight[:, None] * influence
        covariance_matrix = weighted_influence.T @ weighted_influence
        standard_errors = np.sqrt(np.maximum(np.diag(covariance_matrix), 0.0))
        effective_nobs = float(1.0 / np.sum(probability_weight**2))

        for fold in fold_results:
            idx = fold["test_idx"]
            fold["k_weights"] = k_weight[idx]
            fold["moments"] = moments[idx]
            fold["influence_moments"] = influence_moments[idx]
            fold["influence_functions"] = influence[idx]
            # Familiar diagnostics from the original results object.
            fold["theta_x"] = conditional_elasticity[idx]
            fold["alpha_x"] = -(
                k_weight[idx, None]
                * (
                    h_prime[idx]
                    + h[idx] * density_score[idx]
                    + self.psi * estimates[None, :]
                )
            )
            fold["correction_weight"] = fold["alpha_x"]

        return GeneralizedDREAMResults(
            estimates=np.asarray(estimates, dtype=np.float64),
            standard_errors=np.asarray(standard_errors, dtype=np.float64),
            exog_names=list(self.exog_names),
            endog_name=self.endog_names,
            interest_indices=interest_indices,
            variable_types=dict(self.variable_types),
            beta=np.asarray(self.beta[interest_indices], dtype=np.float64),
            ols_results=self.ols_res,
            phi=self.phi,
            psi=self.psi,
            policy_name=self.policy_name,
            n_folds=n_folds,
            nobs=self.nobs,
            effective_nobs=effective_nobs,
            fold_results=fold_results,
            moments=moments,
            influence_moments=influence_moments,
            influence_functions=influence,
            covariance_matrix=covariance_matrix,
            fold_weights=fold_weights,
            k_weights=k_weight,
            log_normalizer=log_normalizer,
            normalizer=normalizer,
            denominator=denominator,
            _is_pandas=self._original_is_pandas,
        )


class JointGeneralizedDREAM(GeneralizedDREAM):
    """Estimate several Generalized-DREAm targets in one cross-fitted system.

    The density score, OLS fold coefficient and sample split are shared by all
    targets.  A separate nuisance regression is fitted once per distinct
    ``phi`` and reused for every requested ``psi``.  Observation-level influence
    functions are stacked before computing the covariance matrix, preserving
    all cross-target and cross-variable covariances.
    """

    DEFAULT_TARGETS: tuple[Target, ...] = (
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        exog: npt.ArrayLike,
        weights: npt.ArrayLike | None = None,
        fixed_effects: list[str] | list[int] | None = None,
        interest: list[str] | list[int] | str | int | None = None,
        *,
        targets: Sequence[Target] = DEFAULT_TARGETS,
        h: Policy | None = None,
        h_derivative: Policy | None = None,
        **kwargs: Any,
    ) -> None:
        normalized_targets: list[Target] = []
        for position, target in enumerate(targets):
            if not isinstance(target, Sequence) or len(target) != 2:
                raise ValueError(
                    f"targets[{position}] must be a two-item (phi, psi) sequence"
                )
            phi = self._finite_scalar(target[0], f"targets[{position}].phi")
            psi = self._finite_scalar(target[1], f"targets[{position}].psi")
            normalized_targets.append((phi, psi))
        if not normalized_targets:
            raise ValueError("At least one (phi, psi) target is required")
        if len(set(normalized_targets)) != len(normalized_targets):
            raise ValueError("targets must not contain duplicates")
        self.targets = tuple(normalized_targets)

        super().__init__(
            endog=endog,
            exog=exog,
            weights=weights,
            fixed_effects=fixed_effects,
            interest=interest,
            phi=self.targets[0][0],
            psi=self.targets[0][1],
            h=h,
            h_derivative=h_derivative,
            **kwargs,
        )

    def fit(
        self,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        m_params: Optional[Mapping[str, Any]] = None,
        density_params: Optional[Mapping[str, Any]] = None,
    ) -> JointGeneralizedDREAMResults:
        """Fit all requested targets and their joint covariance matrix."""
        interest_indices = list(self.interest)
        n_interest = len(interest_indices)
        unique_phis = tuple(dict.fromkeys(phi for phi, _ in self.targets))
        m_configs = {
            phi: self._prepare_m_params(m_params, phi=phi) for phi in unique_phis
        }
        density_model_class, density_config = self._prepare_density_params(
            density_params
        )
        splits = self._make_splits(n_folds, random_state)

        fold_results: list[dict[str, Any]] = []
        for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
            weight_fold = np.ones(self.nobs, dtype=np.float64)
            weight_fold[test_idx] = 1e-10
            if self.weights is not None:
                weight_fold *= self.weights
            print(
                f"Processing joint generalized DREAm fold "
                f"{fold_number}/{n_folds}"
            )
            common = self._process_generalized_fold_common(
                train_idx,
                test_idx,
                weight_fold,
                interest_indices,
                density_model_class,
                density_config,
            )
            phi_results = {
                phi: self._process_phi_fold(
                    common, interest_indices, phi, m_configs[phi]
                )
                for phi in unique_phis
            }
            fold_results.append(
                {
                    **self._public_common_fold(common),
                    "phi_results": phi_results,
                }
            )

        density_score = np.empty((self.nobs, n_interest), dtype=np.float64)
        h = np.empty_like(density_score)
        h_prime = np.empty_like(density_score)
        phi_arrays = {
            phi: {
                "conditional_elasticity": np.empty_like(density_score),
                "log_inner_mean": np.empty(self.nobs, dtype=np.float64),
                "r_phi": np.empty(self.nobs, dtype=np.float64),
            }
            for phi in unique_phis
        }
        for fold in fold_results:
            idx = fold["test_idx"]
            density_score[idx] = fold["density_score"]
            h[idx] = fold["h"]
            h_prime[idx] = fold["h_prime"]
            for phi in unique_phis:
                phi_fold = fold["phi_results"][phi]
                phi_arrays[phi]["conditional_elasticity"][idx] = phi_fold[
                    "conditional_elasticity"
                ]
                phi_arrays[phi]["log_inner_mean"][idx] = phi_fold[
                    "log_inner_mean"
                ]
                phi_arrays[phi]["r_phi"][idx] = phi_fold["r_phi"]

        fold_weights = (
            np.ones(self.nobs, dtype=np.float64)
            if self.weights is None
            else np.asarray(self.weights, dtype=np.float64).copy()
        )
        target_solutions: list[dict[str, Any]] = []
        for phi, psi in self.targets:
            arrays = phi_arrays[phi]
            target_solutions.append(
                self._solve_target(
                    conditional_elasticity=arrays["conditional_elasticity"],
                    density_score=density_score,
                    h=h,
                    h_prime=h_prime,
                    log_inner_mean=arrays["log_inner_mean"],
                    r_phi=arrays["r_phi"],
                    fold_weights=fold_weights,
                    psi=psi,
                )
            )

        estimates = np.vstack(
            [solution["estimates"] for solution in target_solutions]
        )
        moments = np.column_stack(
            [solution["moments"] for solution in target_solutions]
        )
        influence_moments = np.column_stack(
            [solution["influence_moments"] for solution in target_solutions]
        )
        influence_functions = np.column_stack(
            [solution["influence_functions"] for solution in target_solutions]
        )
        probability_weight = fold_weights / fold_weights.sum()
        weighted_influence = probability_weight[:, None] * influence_functions
        covariance_matrix = weighted_influence.T @ weighted_influence
        covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
        standard_errors = np.sqrt(
            np.maximum(np.diag(covariance_matrix), 0.0)
        ).reshape(len(self.targets), n_interest)
        effective_nobs = float(1.0 / np.sum(probability_weight**2))

        k_weights = np.column_stack(
            [solution["k_weights"] for solution in target_solutions]
        )
        log_normalizers = np.asarray(
            [solution["log_normalizer"] for solution in target_solutions],
            dtype=np.float64,
        )
        normalizers = np.asarray(
            [solution["normalizer"] for solution in target_solutions],
            dtype=np.float64,
        )
        denominators = np.asarray(
            [solution["denominator"] for solution in target_solutions],
            dtype=np.float64,
        )

        for fold in fold_results:
            idx = fold["test_idx"]
            fold["target_results"] = {}
            for target_position, ((phi, psi), solution) in enumerate(
                zip(self.targets, target_solutions)
            ):
                arrays = phi_arrays[phi]
                target_estimate = estimates[target_position]
                alpha = -(
                    solution["k_weights"][idx, None]
                    * (
                        h_prime[idx]
                        + h[idx] * density_score[idx]
                        + psi * target_estimate[None, :]
                    )
                )
                column_start = target_position * n_interest
                column_stop = column_start + n_interest
                fold["target_results"][(phi, psi)] = {
                    "k_weights": solution["k_weights"][idx],
                    "moments": moments[idx, column_start:column_stop],
                    "influence_moments": influence_moments[
                        idx, column_start:column_stop
                    ],
                    "influence_functions": influence_functions[
                        idx, column_start:column_stop
                    ],
                    "theta_x": arrays["conditional_elasticity"][idx],
                    "alpha_x": alpha,
                    "correction_weight": alpha,
                }

        return JointGeneralizedDREAMResults(
            targets=self.targets,
            estimates=estimates,
            standard_errors=standard_errors,
            exog_names=list(self.exog_names),
            endog_name=self.endog_names,
            interest_indices=interest_indices,
            variable_types=dict(self.variable_types),
            beta=np.asarray(self.beta[interest_indices], dtype=np.float64),
            ols_results=self.ols_res,
            policy_name=self.policy_name,
            n_folds=n_folds,
            nobs=self.nobs,
            effective_nobs=effective_nobs,
            fold_results=fold_results,
            moments=moments,
            influence_moments=influence_moments,
            influence_functions=influence_functions,
            covariance_matrix=covariance_matrix,
            fold_weights=fold_weights,
            k_weights=k_weights,
            log_normalizers=log_normalizers,
            normalizers=normalizers,
            denominators=denominators,
            _is_pandas=self._original_is_pandas,
        )


class GeneralizedPCAM(DoublyRobustElasticityEstimatorModel):
    r"""Generalised percentage change of aggregate means for binary treatment.

    This implements the GPCAM score in the binary-treatment section of the
    paper.  If ``Q_d`` is the design matrix with the treatment set to ``d``,
    the estimator cross-fits ``beta``, ``m_{phi,d}(W)`` and the propensity
    score, then solves

    .. math::

        E\left[A_1\{1+\psi D R_{\phi,1}/\pi_1(W)\}\right]
        -(1+\theta)^\psi
        E\left[A_0\{1+\psi(1-D)R_{\phi,0}/\pi_0(W)\}\right]=0,

    where ``A_d = exp(psi * Q_d' beta) * m_{phi,d}(W)**(psi/phi)``.
    The continuous limits are used at ``phi=0`` and ``psi=0``.

    Parameters
    ----------
    endog : array_like
        Strictly positive outcome.
    exog : array_like
        Design matrix containing one binary treatment column and the controls.
        Add a constant explicitly when one is required by the log-linear
        working model.
    weights, fixed_effects
        As in :class:`DoublyRobustElasticityEstimatorModel`.
    interest, treatment : str or int, optional
        The 0/1 treatment column.  ``treatment`` is the descriptive spelling;
        ``interest`` is accepted for consistency with GEAM.  Supply exactly
        one of them.
    phi : float, default 1
        Inner power-mean parameter.
    psi : float, default 1
        Outer power-mean parameter.  Zero uses the geometric-mean limit.
    propensity_clip : float, default 0.01
        Cross-fitted propensity predictions are clipped to
        ``[propensity_clip, 1-propensity_clip]``.
    """

    def __init__(
        self,
        endog: npt.ArrayLike,
        exog: npt.ArrayLike,
        weights: npt.ArrayLike | None = None,
        fixed_effects: list[str] | list[int] | None = None,
        interest: list[str] | list[int] | str | int | None = None,
        *,
        treatment: str | int | None = None,
        phi: float = 1.0,
        psi: float = 1.0,
        propensity_clip: float = 0.01,
        **kwargs: Any,
    ) -> None:
        if treatment is not None and interest is not None:
            raise ValueError("Supply treatment or interest, not both")
        selected_treatment = treatment if treatment is not None else interest
        if selected_treatment is None:
            raise ValueError("A binary treatment column must be supplied")

        self.phi = self._finite_scalar(phi, "phi")
        self.psi = self._finite_scalar(psi, "psi")
        self.propensity_clip = self._finite_scalar(
            propensity_clip, "propensity_clip"
        )
        if not 0.0 < self.propensity_clip < 0.5:
            raise ValueError("propensity_clip must lie strictly between 0 and 0.5")

        y = np.asarray(endog, dtype=np.float64).ravel()
        if y.size == 0 or not np.all(np.isfinite(y)) or np.any(y <= 0):
            raise ValueError("endog must contain finite, strictly positive values")
        supplied_weights = (
            None
            if weights is None
            else np.asarray(weights, dtype=np.float64).ravel()
        )

        super().__init__(
            endog=endog,
            exog=exog,
            weights=None,
            fixed_effects=fixed_effects,
            interest=selected_treatment,
            ordinal=None,
            **kwargs,
        )
        if not np.all(np.isfinite(self.exog)):
            raise ValueError("exog must contain only finite values")
        if supplied_weights is not None:
            self._install_weights(supplied_weights)

        treatment_indices = list(self.interest)
        if len(treatment_indices) != 1:
            raise ValueError("GPCAM requires exactly one treatment column")
        self.treatment_index = treatment_indices[0]
        self.treatment_name = self.exog_names[self.treatment_index]
        treatment_values = np.unique(self.exog[:, self.treatment_index])
        if treatment_values.shape != (2,) or not np.array_equal(
            treatment_values, np.array([0.0, 1.0])
        ):
            raise ValueError("The GPCAM treatment must be coded exactly as 0 and 1")
        self.treatment = self.exog[:, self.treatment_index].astype(np.int64)

    @staticmethod
    def _finite_scalar(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _install_weights(self, weights: Array) -> None:
        if weights.shape != (self.nobs,):
            raise ValueError(f"weights must have length {self.nobs}")
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError(
                "weights must be finite, non-negative, and have positive sum"
            )

        self.weights = weights
        if self.fixed_effects is None:
            self.ols_res = sm.WLS(
                np.log(self.endog), self.exog, weights=weights
            ).fit()
        else:
            y_dm, x_dm = _apply_fixed_effects(
                np.log(self.endog),
                self.exog,
                algorithm=self.fixed_effects,
                weights=weights.reshape(-1, 1),
            )
            self.temp_exog_demeaned = x_dm
            self.ols_res = sm.WLS(y_dm, x_dm, weights=weights).fit()
        self.beta = np.asarray(self.ols_res.params, dtype=np.float64)

    def _prepare_m_params(
        self, m_params: Optional[Mapping[str, Any]]
    ) -> dict[str, Any]:
        supplied = deepcopy(dict(m_params)) if m_params is not None else None
        parsed = self._parse_m_params(supplied)
        user_set_activation = bool(
            m_params is not None
            and isinstance(m_params.get("arch_params"), Mapping)
            and "output_activation" in m_params["arch_params"]
        )
        if not user_set_activation:
            parsed["arch_params"]["output_activation"] = "identity"
        if self.phi != 0.0:
            parsed["arch_params"].setdefault("positive_log_link", True)
            parsed["arch_params"].setdefault("log_link_max_abs", 3.0)
        return parsed

    def _prepare_propensity_params(
        self, propensity_params: Optional[Mapping[str, Any]]
    ) -> dict[str, Any]:
        supplied = (
            deepcopy(dict(propensity_params))
            if propensity_params is not None
            else None
        )
        parsed = self._parse_m_params(supplied)
        user_set_activation = bool(
            propensity_params is not None
            and isinstance(propensity_params.get("arch_params"), Mapping)
            and "output_activation" in propensity_params["arch_params"]
        )
        if not user_set_activation:
            parsed["arch_params"]["output_activation"] = "sigmoid"
        parsed["arch_params"].pop("positive_log_link", None)
        parsed["arch_params"].pop("positive_floor", None)
        return parsed

    def _make_splits(
        self, n_folds: int, random_state: Optional[int]
    ) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        arm_counts = np.bincount(self.treatment, minlength=2)
        if (
            not isinstance(n_folds, int)
            or n_folds < 2
            or n_folds > int(np.min(arm_counts))
        ):
            raise ValueError(
                "n_folds must be an integer between 2 and the size of the "
                "smaller treatment arm"
            )
        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
        return [
            (np.asarray(train), np.asarray(test))
            for train, test in splitter.split(self.exog, self.treatment)
        ]

    @staticmethod
    def _coerce_prediction(
        prediction: npt.ArrayLike, n: int, name: str
    ) -> Array:
        result = np.asarray(prediction, dtype=np.float64).reshape(-1)
        if result.shape != (n,):
            raise ValueError(f"{name} returned {result.shape}; expected ({n},)")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} returned non-finite values")
        return result

    def _process_gpcam_fold(
        self,
        train_idx: npt.NDArray[np.int64],
        test_idx: npt.NDArray[np.int64],
        weight_fold: Array,
        m_params: dict[str, Any],
        propensity_params: dict[str, Any],
    ) -> dict[str, Any]:
        # Same fold OLS and fixed-effect handling as the continuous estimator
        ols, beta, residual_train, residual_test, _, fe_test = (
            GeneralizedDREAM._fold_ols_residuals(self, train_idx, test_idx, weight_fold)
        )

        x_test = self.exog[test_idx]
        x_aug_train = self.exog_augmented[train_idx]
        x_aug_test = self.exog_augmented[test_idx]
        d_train = self.treatment[train_idx]
        d_test = self.treatment[test_idx]

        if self.phi == 0.0:
            nuisance_target = residual_train
            transformed_test = residual_test
        else:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                nuisance_target = np.exp(self.phi * residual_train)
                transformed_test = np.exp(self.phi * residual_test)
            if (
                not np.all(np.isfinite(nuisance_target))
                or not np.all(np.isfinite(transformed_test))
                or np.any(nuisance_target <= 0)
                or np.any(transformed_test <= 0)
            ):
                raise FloatingPointError(
                    "exp(phi * residual) overflowed or underflowed; rescale the "
                    "outcome or use a less extreme phi"
                )

        m_predictions = np.empty((len(test_idx), 2), dtype=np.float64)
        log_inner_means = np.empty_like(m_predictions)
        r_phi = np.zeros_like(m_predictions)
        m_results: dict[int, Any] = {}
        m_model_class = m_params.get("model_class", NNModelNuisance)
        for arm in (0, 1):
            arm_train = d_train == arm
            if not np.any(arm_train):
                raise ValueError(f"Treatment arm {arm} is absent from a training fold")
            m_model = m_model_class(
                variable_types=self.variable_types,
                **m_params["arch_params"],
            )
            m_result = m_model.fit(
                x_aug_train[arm_train],
                nuisance_target[arm_train],
                **m_params["fit_params"],
            )
            qd_aug_test = x_aug_test.copy()
            qd_aug_test[:, self.treatment_index] = float(arm)
            m_hat = self._coerce_prediction(
                m_result.predict(qd_aug_test), len(test_idx), f"m_phi_{arm}"
            )
            m_predictions[:, arm] = m_hat
            qd_test = x_test.copy()
            qd_test[:, self.treatment_index] = float(arm)
            # Fixed-effect component added back so M_phi is the power mean of y itself
            linear_predictor = qd_test @ beta + fe_test
            observed_arm = d_test == arm
            if self.phi == 0.0:
                log_inner_means[:, arm] = linear_predictor + m_hat
                r_phi[observed_arm, arm] = (
                    transformed_test[observed_arm] - m_hat[observed_arm]
                )
            else:
                if np.any(m_hat <= 0):
                    raise ValueError(
                        f"m_phi_{arm} must be strictly positive; use a "
                        "positive-output nuisance model"
                    )
                log_inner_means[:, arm] = (
                    linear_predictor + np.log(m_hat) / self.phi
                )
                r_phi[observed_arm, arm] = (
                    transformed_test[observed_arm] - m_hat[observed_arm]
                ) / (self.phi * m_hat[observed_arm])
            m_results[arm] = m_result

        propensity_model_class = propensity_params.get(
            "model_class", NNModelNuisance
        )
        propensity_model = propensity_model_class(
            variable_types=self.variable_types,
            **propensity_params["arch_params"],
        )
        # The treatment column is fixed before fitting and predicting so the
        # propensity model only receives W (plus any encoded fixed effects).
        w_aug_train = x_aug_train.copy()
        w_aug_test = x_aug_test.copy()
        w_aug_train[:, self.treatment_index] = 0.0
        w_aug_test[:, self.treatment_index] = 0.0
        propensity_results = propensity_model.fit(
            w_aug_train,
            d_train.astype(np.float64),
            **propensity_params["fit_params"],
        )
        propensity_raw = self._coerce_prediction(
            propensity_results.predict(w_aug_test),
            len(test_idx),
            "propensity model",
        )
        if np.any((propensity_raw < 0.0) | (propensity_raw > 1.0)):
            raise ValueError("Propensity predictions must lie in [0, 1]")
        propensity = np.clip(
            propensity_raw, self.propensity_clip, 1.0 - self.propensity_clip
        )

        return {
            "beta": beta,
            "ols_results": ols,
            "m_results": m_results,
            "propensity_results": propensity_results,
            "propensity_raw": propensity_raw,
            "propensity": propensity,
            "m_phi": m_predictions,
            "log_inner_means": log_inner_means,
            "r_phi": r_phi,
            "train_idx": train_idx,
            "test_idx": test_idx,
        }

    @staticmethod
    def _solve_target(
        *,
        treatment: Array,
        propensity: Array,
        log_inner_means: Array,
        r_phi: Array,
        fold_weights: Array,
        psi: float,
    ) -> dict[str, Any]:
        treatment = np.asarray(treatment, dtype=np.float64).reshape(-1)
        propensity = np.asarray(propensity, dtype=np.float64).reshape(-1)
        log_inner_means = np.asarray(log_inner_means, dtype=np.float64)
        r_phi = np.asarray(r_phi, dtype=np.float64)
        fold_weights = np.asarray(fold_weights, dtype=np.float64).reshape(-1)
        nobs = len(treatment)
        if (
            propensity.shape != (nobs,)
            or log_inner_means.shape != (nobs, 2)
            or r_phi.shape != (nobs, 2)
            or fold_weights.shape != (nobs,)
        ):
            raise ValueError("GPCAM score inputs have incompatible shapes")
        if (
            not np.all(np.isfinite(propensity))
            or np.any((propensity <= 0.0) | (propensity >= 1.0))
            or not np.all(np.isfinite(log_inner_means))
            or not np.all(np.isfinite(r_phi))
            or not np.all(np.isfinite(fold_weights))
            or np.any(fold_weights < 0.0)
            or fold_weights.sum() <= 0.0
        ):
            raise ValueError("GPCAM score inputs must be finite with valid weights")

        probability_weight = fold_weights / fold_weights.sum()
        d1 = treatment
        d0 = 1.0 - treatment
        pi1 = propensity
        pi0 = 1.0 - propensity

        if psi == 0.0:
            arm_components = np.column_stack(
                [
                    log_inner_means[:, 0] + d0 * r_phi[:, 0] / pi0,
                    log_inner_means[:, 1] + d1 * r_phi[:, 1] / pi1,
                ]
            )
            arm_log_means = np.average(
                arm_components, axis=0, weights=fold_weights
            )
            log_growth_factor = float(arm_log_means[1] - arm_log_means[0])
            if log_growth_factor > np.log(np.finfo(np.float64).max):
                raise FloatingPointError("The estimated GPCAM target overflowed")
            growth_factor = float(np.exp(log_growth_factor))
            estimate = growth_factor - 1.0
            moments = (
                arm_components[:, 1]
                - arm_components[:, 0]
                - log_growth_factor
            )
            score_derivative = -1.0 / growth_factor
            influence = growth_factor * moments
            with np.errstate(over="ignore"):
                arm_means = np.exp(arm_log_means)
            score_scale_log = 0.0
        else:
            log_power_weights = psi * log_inner_means
            max_abs_log_weight = float(np.max(np.abs(log_power_weights)))
            # Preserve the paper's score on ordinary data.  Only introduce a
            # common scale when exponentiation would be numerically fragile.
            score_scale_log = (
                float(np.max(log_power_weights))
                if max_abs_log_weight > 500.0
                else 0.0
            )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                power_weights = np.exp(log_power_weights - score_scale_log)
            arm_components = np.column_stack(
                [
                    power_weights[:, 0]
                    * (1.0 + psi * d0 * r_phi[:, 0] / pi0),
                    power_weights[:, 1]
                    * (1.0 + psi * d1 * r_phi[:, 1] / pi1),
                ]
            )
            if not np.all(np.isfinite(arm_components)):
                raise FloatingPointError("The GPCAM score contains non-finite values")
            scaled_arm_means = np.average(
                arm_components, axis=0, weights=fold_weights
            )
            if np.any(~np.isfinite(scaled_arm_means)) or np.any(
                scaled_arm_means <= 0.0
            ):
                raise FloatingPointError(
                    "The augmented GPCAM arm means must be strictly positive"
                )
            arm_log_means = score_scale_log + np.log(scaled_arm_means)
            log_ratio = float(arm_log_means[1] - arm_log_means[0])
            log_growth_factor = log_ratio / psi
            if log_growth_factor > np.log(np.finfo(np.float64).max):
                raise FloatingPointError("The estimated GPCAM target overflowed")
            growth_factor = float(np.exp(log_growth_factor))
            estimate = growth_factor - 1.0
            ratio = float(scaled_arm_means[1] / scaled_arm_means[0])
            moments = arm_components[:, 1] - ratio * arm_components[:, 0]
            score_derivative = float(
                -psi * scaled_arm_means[1] / growth_factor
            )
            if (
                not np.isfinite(score_derivative)
                or abs(score_derivative) <= np.sqrt(np.finfo(float).eps)
            ):
                raise FloatingPointError(
                    "The sample derivative of the GPCAM score is numerically zero"
                )
            influence = -moments / score_derivative
            with np.errstate(over="ignore"):
                arm_means = np.exp(arm_log_means)

        influence -= float(np.sum(probability_weight * influence))
        return {
            "estimate": float(estimate),
            "moments": np.asarray(moments, dtype=np.float64),
            "influence_functions": np.asarray(influence, dtype=np.float64),
            "arm_components": np.asarray(arm_components, dtype=np.float64),
            "arm_means": np.asarray(arm_means, dtype=np.float64),
            "log_arm_means": np.asarray(arm_log_means, dtype=np.float64),
            "score_derivative": float(score_derivative),
            "score_scale_log": float(score_scale_log),
        }

    def fit(
        self,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        m_params: Optional[Mapping[str, Any]] = None,
        propensity_params: Optional[Mapping[str, Any]] = None,
        *,
        pi_params: Optional[Mapping[str, Any]] = None,
    ) -> GPCAMResults:
        """Fit the cross-fitted binary-treatment GPCAM estimator.

        ``m_params`` configures both arm-specific residual-moment regressions.
        ``propensity_params`` configures the propensity regression.  Both use
        the package's ``arch_params``/``fit_params``/``model_class`` convention;
        ``pi_params`` is accepted as an alias for ``propensity_params``.
        """
        if propensity_params is not None and pi_params is not None:
            raise ValueError("Supply propensity_params or pi_params, not both")
        propensity_params = (
            propensity_params if propensity_params is not None else pi_params
        )
        m_config = self._prepare_m_params(m_params)
        propensity_config = self._prepare_propensity_params(propensity_params)
        splits = self._make_splits(n_folds, random_state)

        fold_results: list[dict[str, Any]] = []
        for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
            weight_fold = np.ones(self.nobs, dtype=np.float64)
            weight_fold[test_idx] = 1e-10
            if self.weights is not None:
                weight_fold *= self.weights
            print(f"Processing GPCAM fold {fold_number}/{n_folds}")
            fold_results.append(
                self._process_gpcam_fold(
                    train_idx,
                    test_idx,
                    weight_fold,
                    m_config,
                    propensity_config,
                )
            )

        propensity = np.empty(self.nobs, dtype=np.float64)
        log_inner_means = np.empty((self.nobs, 2), dtype=np.float64)
        r_phi = np.empty_like(log_inner_means)
        for fold in fold_results:
            idx = fold["test_idx"]
            propensity[idx] = fold["propensity"]
            log_inner_means[idx] = fold["log_inner_means"]
            r_phi[idx] = fold["r_phi"]

        fold_weights = (
            np.ones(self.nobs, dtype=np.float64)
            if self.weights is None
            else np.asarray(self.weights, dtype=np.float64).copy()
        )
        solution = self._solve_target(
            treatment=self.treatment.astype(np.float64),
            propensity=propensity,
            log_inner_means=log_inner_means,
            r_phi=r_phi,
            fold_weights=fold_weights,
            psi=self.psi,
        )
        probability_weight = fold_weights / fold_weights.sum()
        influence = solution["influence_functions"]
        variance = float(np.sum((probability_weight * influence) ** 2))
        covariance_matrix = np.asarray([[variance]], dtype=np.float64)
        standard_error = float(np.sqrt(max(variance, 0.0)))
        effective_nobs = float(1.0 / np.sum(probability_weight**2))

        for fold in fold_results:
            idx = fold["test_idx"]
            fold["arm_components"] = solution["arm_components"][idx]
            fold["moments"] = solution["moments"][idx]
            fold["influence_functions"] = influence[idx]

        return GPCAMResults(
            estimate=solution["estimate"],
            standard_error=standard_error,
            phi=self.phi,
            psi=self.psi,
            treatment_name=self.treatment_name,
            treatment_index=self.treatment_index,
            exog_names=list(self.exog_names),
            endog_name=self.endog_names,
            beta=np.asarray(self.beta, dtype=np.float64),
            ols_results=self.ols_res,
            n_folds=n_folds,
            nobs=self.nobs,
            effective_nobs=effective_nobs,
            fold_results=fold_results,
            moments=solution["moments"],
            influence_functions=influence,
            covariance_matrix=covariance_matrix,
            fold_weights=fold_weights,
            arm_means=solution["arm_means"],
            log_arm_means=solution["log_arm_means"],
            score_derivative=solution["score_derivative"],
            score_scale_log=solution["score_scale_log"],
        )


# Paper terminology and explicit long-form aliases.
GEAM = GeneralizedDREAM
JGEAM = JointGeneralizedDREAM
GPCAM = GeneralizedPCAM
GeneralisedDREAM = GeneralizedDREAM
JointGeneralisedDREAM = JointGeneralizedDREAM
GeneralizedDoublyRobustElasticityEstimatorModel = GeneralizedDREAM
GeneralisedPCAM = GeneralizedPCAM
GeneralizedPercentageChangeAggregateMeansEstimator = GeneralizedPCAM
GEAMResults = GeneralizedDREAMResults
JGEAMResults = JointGeneralizedDREAMResults


__all__ = [
    "GEAM",
    "GEAMResults",
    "GPCAM",
    "GPCAMResults",
    "GeneralisedPCAM",
    "JGEAM",
    "JGEAMResults",
    "GeneralisedDREAM",
    "GeneralizedDREAM",
    "GeneralizedDREAMResults",
    "GeneralizedDoublyRobustElasticityEstimatorModel",
    "GeneralizedPCAM",
    "GeneralizedPercentageChangeAggregateMeansEstimator",
    "JointGeneralisedDREAM",
    "JointGeneralizedDREAM",
    "JointGeneralizedDREAMResults",
]
