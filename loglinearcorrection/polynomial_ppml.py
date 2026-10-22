"""
PPML-polynomial elasticity estimator (MGF-decomposition estimator).

Estimates the conditional mean by Poisson pseudo-maximum-likelihood (PPML) with a
polynomial in the interest variable,

    E[y | x] = exp(g0 + g1 * x + g2 * x**2 + ... + gd * x**d  +  controls + FE),

reads the per-observation response off the derivative of the linear predictor, and
**averages over the observed data** to get the average elasticity.

Because the response is differentiated w.r.t. whatever column is passed (exactly as
DREEM differentiates w.r.t. its exog column), the reported quantity is in the same
units as DREEM: pass ``log x`` for an elasticity, or ``x`` for a semi-elasticity.

Per-observation response and its average:

    e_i  = sum_{p=1}^d  p * g_p * x_i**(p-1)
    eps  = (1/n) sum_i e_i  =  g_bar' @ gamma_poly,
           g_bar_p = (1/n) sum_i p * x_i**(p-1)

Delta-method SE (conditioning on the observed x, i.e. treating the weights g_bar as
fixed -- the same convention as paper/code/application_distance.py:run_ppml_quad):

    SE = sqrt( g_bar' @ V_poly @ g_bar ),

with ``V_poly`` the sub-block of the PPML coefficient covariance for the polynomial
terms of that variable.

A specification test for the constant-elasticity (log-OLS) assumption is provided via
:meth:`PolyPPMLResults.curvature_test`: by default a two-sided test on the ``x**2``
coefficient; with ``joint=True`` (and degree > 2) a joint Wald test that all order-2+
coefficients are zero.

Backend: ``statsmodels`` GLM Poisson when there are no fixed effects (named, robust
``cov_params`` -- matches the existing application code), and ``pyfixest.fepois`` when
fixed effects are supplied.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from .utils import _initialize_fixed_effects, _apply_fixed_effects, _redundant_columns


def _poly_colname(var: str, power: int) -> str:
    """Safe design-matrix column name for ``var`` raised to ``power``."""
    return var if power == 1 else f"{var}__pow{power}"


def is_continuous(values: npt.ArrayLike) -> bool:
    """Continuous in the DREEM sense: not exactly 2 unique values (i.e. not binary).

    Mirrors ``loglinearcorrection.utils._detect_variable_types`` -- a column with
    exactly two distinct values is binary; everything else is treated as continuous.
    A polynomial in a binary variable is degenerate (x**2 == x), so the PPML-polynomial
    estimator only applies to continuous variables.
    """
    return len(np.unique(np.asarray(values))) != 2


class PolynomialPPMLElasticity:
    """PPML with a polynomial in the interest variable(s); average elasticity by
    differentiating the fitted linear predictor and averaging over the sample.

    Parameters
    ----------
    endog : array_like
        Outcome in levels (PPML models ``E[y|x]`` directly, not ``log y``).
    exog : array_like
        Regressors. May be a pandas DataFrame/Series or a numpy array. Interest
        variables are expanded into a polynomial; all other (non-FE) columns enter
        linearly as controls.
    interest : str | int | sequence
        Variable name(s) (if ``exog`` is a DataFrame) or column index/indices to
        expand into a polynomial and report the elasticity for.
    degree : int, default 2
        Polynomial degree. ``degree=2`` reproduces the quadratic special case.
    fixed_effects : sequence of str or int, optional
        Columns of ``exog`` to absorb as fixed effects (uses ``pyfixest.fepois``).
    cov_type : str, default "HC3"
        Covariance type for the no-fixed-effects ``statsmodels`` path. Ignored on the
        fixed-effects path (which uses ``pyfixest`` heteroskedasticity-robust vcov).
    """

    def __init__(
        self,
        endog: npt.ArrayLike,
        exog: npt.ArrayLike,
        interest: Union[str, int, Sequence[Union[str, int]]],
        degree: int = 2,
        fixed_effects: Optional[Sequence[Union[str, int]]] = None,
        cov_type: str = "HC3",
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = int(degree)
        self.cov_type = cov_type

        # ---- names ----------------------------------------------------------
        self.endog_name = (
            endog.name if isinstance(endog, pd.Series) and endog.name is not None
            else endog.columns[0] if isinstance(endog, pd.DataFrame) and endog.columns[0] != 0
            else "y"
        )
        # Preserve original dtypes -- fixed-effect (and other) columns may be
        # categorical/string; only the numeric regressors get coerced to float later.
        if isinstance(exog, pd.DataFrame):
            exog_names = list(exog.columns)
            df_in = exog.reset_index(drop=True)
        elif isinstance(exog, pd.Series):
            exog_names = [exog.name]
            df_in = exog.to_frame().reset_index(drop=True)
        else:
            arr = np.asarray(exog)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            exog_names = [f"x{i + 1}" for i in range(arr.shape[1])]
            df_in = pd.DataFrame(arr, columns=exog_names)

        # ---- resolve fixed effects -----------------------------------------
        fe_names = self._resolve(fixed_effects, exog_names) if fixed_effects is not None else []
        self.fe_names = fe_names

        # ---- resolve interest ----------------------------------------------
        if isinstance(interest, (str, int, np.integer)):
            interest = [interest]
        self.interest_names = self._resolve(interest, exog_names)
        if any(v in fe_names for v in self.interest_names):
            raise ValueError("interest variables cannot also be fixed effects")

        # ---- store ----------------------------------------------------------
        self.endog = np.asarray(endog, dtype=np.float64).ravel()
        self.exog_names = exog_names
        self._df = df_in
        self.control_names = [
            v for v in exog_names if v not in self.interest_names and v not in fe_names
        ]

        # Numeric interest columns (also validates they are numeric) and the
        # continuous-only guard: a polynomial in a binary variable is degenerate.
        self._interest_x: dict[str, npt.NDArray[np.float64]] = {}
        for var in self.interest_names:
            x = pd.to_numeric(self._df[var]).to_numpy(dtype=np.float64)
            if not is_continuous(x):
                raise ValueError(
                    f"interest variable {var!r} is binary; PPML-polynomial requires "
                    f"continuous interest variables"
                )
            self._interest_x[var] = x

        self.fit_ = None
        self.coefs: Optional[pd.Series] = None
        self.vcov: Optional[pd.DataFrame] = None

    @staticmethod
    def _resolve(spec: Sequence[Union[str, int]], names: list[str]) -> list[str]:
        out = []
        for s in spec:
            if isinstance(s, str):
                out.append(s)
            else:
                out.append(names[int(s)])
        return out

    def _build_design(self) -> tuple[pd.DataFrame, list[str]]:
        """Design frame (poly powers + controls + FE + outcome) and the regressor names.

        On the fixed-effects path, columns that are absorbed by the fixed effects (no
        within-group variation after demeaning) are dropped using DREEM's exact
        redundancy machinery (``_apply_fixed_effects`` -> ``_redundant_columns``). This
        removes the intercept/``const`` and any time-invariant controls -- which would
        otherwise make the PPML design singular -- and keeps the surviving control set
        the same as the DREEM replication. (Mutual collinearity among non-absorbed
        controls is left to pyfixest, which drops it automatically.)
        """
        poly_names: list[str] = []
        poly_cols: list[npt.NDArray[np.float64]] = []
        for var in self.interest_names:
            x = self._interest_x[var]
            for p in range(1, self.degree + 1):
                poly_names.append(_poly_colname(var, p))
                poly_cols.append(x ** p)

        ctrl_names = list(self.control_names)
        ctrl_cols = [
            pd.to_numeric(self._df[c]).to_numpy(dtype=np.float64) for c in ctrl_names
        ]

        if self.fe_names:
            # Demean the full design by the fixed effects and find absorbed columns.
            design = np.column_stack(poly_cols + ctrl_cols) if ctrl_cols \
                else np.column_stack(poly_cols)
            _, design_dm = _apply_fixed_effects(self.endog, design, self._fe_algorithm())
            redundant = set(_redundant_columns(design_dm).tolist())

            n_poly = len(poly_names)
            absorbed_poly = [poly_names[i] for i in redundant if i < n_poly]
            if absorbed_poly:
                raise ValueError(
                    f"interest polynomial term(s) {absorbed_poly} are absorbed by the "
                    f"fixed effects (no within-group variation); cannot estimate"
                )
            keep_ctrl = [j for j in range(len(ctrl_names)) if (n_poly + j) not in redundant]
            ctrl_names = [ctrl_names[j] for j in keep_ctrl]
            ctrl_cols = [ctrl_cols[j] for j in keep_ctrl]

        # Assemble the whole design at once (avoids pandas fragmentation warnings).
        reg_names = list(poly_names) + list(ctrl_names)
        data = {name: col for name, col in zip(poly_names, poly_cols)}
        data.update({name: col for name, col in zip(ctrl_names, ctrl_cols)})
        data[self.endog_name] = self.endog
        for fe in self.fe_names:
            data[fe] = self._df[fe].to_numpy()  # keep original dtype (categorical OK)
        df = pd.DataFrame(data, index=self._df.index)
        return df, reg_names

    def _fe_algorithm(self):
        """pyhdfe demeaning algorithm built from the (factorized) fixed-effect columns."""
        codes = np.column_stack(
            [pd.factorize(self._df[fe])[0] for fe in self.fe_names]
        ).astype(np.float64)
        return _initialize_fixed_effects(codes)

    def fit(self) -> "PolyPPMLResults":
        df, reg_names = self._build_design()

        if self.fe_names:
            import pyfixest

            rhs = " + ".join(reg_names)
            fe_part = " + ".join(self.fe_names)
            formula = f"{self.endog_name} ~ {rhs} | {fe_part}"
            m = pyfixest.fepois(formula, data=df, vcov="hetero", drop_intercept=True)
            coefs = m.coef()
            vcov = pd.DataFrame(m._vcov, index=m._coefnames, columns=m._coefnames)
            self.fit_ = m
        else:
            import statsmodels.api as sm

            X = sm.add_constant(df[reg_names], has_constant="skip")
            m = sm.GLM(df[self.endog_name], X, family=sm.families.Poisson()).fit(
                cov_type=self.cov_type
            )
            coefs = m.params
            vcov = m.cov_params()
            self.fit_ = m

        self.coefs = coefs
        self.vcov = vcov
        return PolyPPMLResults(self)


class PolyPPMLResults:
    """Results from :class:`PolynomialPPMLElasticity`."""

    def __init__(self, model: PolynomialPPMLElasticity) -> None:
        self._model = model
        self.coefs = model.coefs
        self.vcov = model.vcov
        self.fit_ = model.fit_
        self.degree = model.degree

    # -- internals -----------------------------------------------------------
    def _check_var(self, var: str) -> str:
        if var not in self._model.interest_names:
            raise ValueError(
                f"{var!r} is not an interest variable; choose from "
                f"{self._model.interest_names}"
            )
        return var

    def _poly_names(self, var: str) -> list[str]:
        return [_poly_colname(var, p) for p in range(1, self.degree + 1)]

    def _gamma(self, var: str) -> npt.NDArray[np.float64]:
        return np.array([self.coefs[n] for n in self._poly_names(var)], dtype=np.float64)

    def _x(self, var: str) -> npt.NDArray[np.float64]:
        return self._model._interest_x[var]

    # -- public API ----------------------------------------------------------
    def per_obs_elasticity(self, var: str) -> npt.NDArray[np.float64]:
        """Per-observation response e_i = sum_p p * g_p * x_i**(p-1)."""
        var = self._check_var(var)
        x = self._x(var)
        gamma = self._gamma(var)
        e = np.zeros_like(x)
        for p in range(1, self.degree + 1):
            e += p * gamma[p - 1] * x ** (p - 1)
        return e

    def avg_elasticity(self, var: str) -> tuple[float, float]:
        """Average elasticity ``g_bar' @ gamma`` and its delta-method SE."""
        var = self._check_var(var)
        x = self._x(var)
        g_bar = np.array(
            [np.mean(p * x ** (p - 1)) for p in range(1, self.degree + 1)],
            dtype=np.float64,
        )
        gamma = self._gamma(var)
        names = self._poly_names(var)
        V = self.vcov.loc[names, names].to_numpy()
        est = float(g_bar @ gamma)
        se = float(np.sqrt(g_bar @ V @ g_bar))
        return est, se

    def curvature_test(self, var: str, joint: bool = False) -> tuple[float, float]:
        """Specification test for non-constant elasticity.

        Default (``joint=False``): two-sided z-test on the ``x**2`` coefficient.
        ``joint=True``: Wald chi2 test that all order-2+ coefficients are zero
        (df = degree - 1). With ``degree == 2`` the two coincide (chi2(1) = z**2).
        """
        var = self._check_var(var)
        if self.degree < 2:
            raise ValueError("curvature_test requires degree >= 2")

        if not joint:
            name = _poly_colname(var, 2)
            b = float(self.coefs[name])
            se = float(np.sqrt(self.vcov.loc[name, name]))
            z = b / se
            p = 2.0 * stats.norm.sf(abs(z))
            return z, p

        names = [_poly_colname(var, p) for p in range(2, self.degree + 1)]
        gamma_high = np.array([self.coefs[n] for n in names], dtype=np.float64)
        V_high = self.vcov.loc[names, names].to_numpy()
        stat = float(gamma_high @ np.linalg.inv(V_high) @ gamma_high)
        df = len(names)
        p = float(stats.chi2.sf(stat, df))
        return stat, p

    def summary(self) -> None:
        for var in self._model.interest_names:
            est, se = self.avg_elasticity(var)
            cz, cp = self.curvature_test(var)
            print(
                f"{var}: avg elasticity = {est: .4f} (se {se:.4f}); "
                f"curvature z = {cz:.3f}, p = {cp:.4f}"
            )
