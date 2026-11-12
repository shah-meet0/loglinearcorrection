r"""Reading (b) panel estimator: elasticity aggregates under panel scale-only confounding.

Setting (``docs/readingb.md`` has the derivations).  Units ``i`` are observed in
several rows ``t``.  Potential outcomes are ``Y_it(x) = C_i * Ytilde_it(x)`` with a
unit scale ``C_i > 0`` whose law may depend arbitrarily on the unit's treatment
and covariate history, and ``Ytilde_it(.)`` independent of ``(C_i, X_i, W_{i,-t})``
given ``W_it``.  The target is

.. math::

    \theta_{\phi,\psi}(h) = E[\omega h \epsilon_\phi] / E[\omega],\qquad
    \omega = (\kappa_\phi(W) m_\phi(X, W))^\psi = M_\phi(Y(x) | W = w)^\psi,

with ``ell_phi = log m_phi`` identified within units, ``epsilon_phi = d ell_phi / d log x``
and ``kappa_phi(w) = M_phi(C | W = w)``.  This is the paper's
``theta_{phi,psi}(h)`` with ``W`` as the cells and the unit effect treated as part of
the unit's type ("reading (b)").

Estimation, per ``phi``:

1. *Within fit.*  ``phi = 0``: FE-OLS of ``log Y``; ``phi != 0``: FE-Poisson of
   ``Y^phi`` (equivalently the multinomial within likelihood).  The default sieve is
   ``ell(x, w) = sum_k b_k(w) s~^k`` with ``s~`` the standardised interest regressor,
   ``b_k(w)`` linear in the controls ``W`` (and saturated in user ``cells``), plus
   main effects of ``W`` and all fixed effects (the unit effect and every other
   fixed effect, e.g. time) absorbed.  With ``degree=1`` and no interactions this is
   the package's log-OLS first stage.  Any nuisance class with the
   ``FESplineModelNuisance`` interface can be plugged in instead.
2. *Cell power mean.*  ``kappa_phi`` is a GLM of the deflated outcome
   ``(Y e^{-ell})^phi`` on cell fixed effects plus the controls (Poisson for
   ``phi != 0``, OLS of the log for ``phi = 0``).  The kappa cells are the user cells
   crossed with every non-unit fixed effect, so additive fixed effects in ``ell``
   cancel from ``kappa e^ell``.
3. *Plug-in* ``theta~ = sum w omega h eps / sum w omega``.
4. *Debiased one-step* with the correction for kappa (the ``gamma`` term) and a Riesz
   representer ``alpha`` for the within fit, learned either automatically in a sieve
   (``riesz='automatic'``) or from the explicit Neumann-series formula
   (``riesz='explicit'``).  Influence functions are per row and are clustered by
   unit (or a coarsening of it) for inference.

``n_folds=1`` (default) uses every row for the nuisances and the score, following
Chen and Chiang (2026, arXiv:2602.11333): with an orthogonal score, nuisances in a
VC-type (low-complexity) class converging faster than ``n^{-1/4}``, inference on the
full-sample estimator is valid without sample splitting.  ``n_folds > 1`` cross-fits
by unit.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from .hypothesis import influence_covariance

Array = npt.NDArray[np.float64]
Target = tuple[float, float]

DEFAULT_TARGETS: tuple[Target, ...] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


# --------------------------------------------------------------------------- linear algebra helpers


def _codes(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    return pd.factorize(np.asarray(values).reshape(-1), sort=True)[0].astype(np.int64)


def _combine_codes(columns: Sequence[npt.ArrayLike], n: int) -> npt.NDArray[np.int64]:
    """One code per distinct row of ``columns`` (all zeros without columns)."""
    if not columns:
        return np.zeros(n, dtype=np.int64)
    frame = pd.DataFrame({j: _codes(c) for j, c in enumerate(columns)})
    return frame.groupby(list(frame.columns), sort=True).ngroup().to_numpy(np.int64)


def _group_mean(v: Array, codes: npt.NDArray[np.int64], w: Array) -> Array:
    """Weighted group means of the columns of ``v`` broadcast back to rows."""
    v2 = v.reshape(len(codes), -1)
    k = int(codes.max()) + 1 if len(codes) else 0
    den = np.bincount(codes, weights=w, minlength=k)
    out = np.empty_like(v2)
    for j in range(v2.shape[1]):
        num = np.bincount(codes, weights=w * v2[:, j], minlength=k)
        means = np.divide(num, den, out=np.zeros(k), where=den > 0)
        out[:, j] = means[codes]
    return out.reshape(v.shape)


class _Demeaner:
    """Weighted residualisation on one or several fixed-effect dimensions.

    One dimension is solved exactly with group means; several use pyhdfe's
    alternating projections.  Weights may change between calls.
    """

    def __init__(self, fe: npt.NDArray[np.int64] | None):
        if fe is None or fe.size == 0 or fe.shape[1] == 0:
            self.fe = None
            self.algorithm = None
            return
        self.fe = np.asarray(fe, dtype=np.int64)
        if self.fe.shape[1] == 1:
            self.algorithm = None
        else:
            import pyhdfe

            self.algorithm = pyhdfe.create(
                self.fe, drop_singletons=False, compute_degrees=False,
                residualize_method="map", options={"tol": 1e-12, "iteration_limit": 100_000},
            )

    def __call__(self, m: Array, w: Array) -> Array:
        m = np.asarray(m, dtype=np.float64)
        if self.fe is None:
            return m - np.average(m, axis=0, weights=w)
        if self.algorithm is None:
            return m - _group_mean(m, self.fe[:, 0], w)
        flat = m.reshape(len(w), -1)
        out = self.algorithm.residualize(flat, weights=(w / w.mean()).reshape(-1, 1))
        return np.asarray(out, dtype=np.float64).reshape(m.shape)


def _lstsq(xs: Array, ys: Array, w: Array, rcond: float = 1e-10) -> tuple[Array, npt.NDArray[np.bool_]]:
    """Weighted least squares with numerically zero columns dropped; returns
    coefficients (zero for dropped columns) and the kept mask."""
    p = xs.shape[1]
    if p == 0:
        return np.zeros((0,) + ys.shape[1:]), np.zeros(0, dtype=bool)
    sw = np.sqrt(w)
    scale = np.sqrt(np.average(xs ** 2, axis=0, weights=w))
    keep = scale > 1e-9 * max(1.0, float(np.max(scale)) if p else 1.0)
    coef = np.zeros((p,) + ys.shape[1:])
    if keep.any():
        a = xs[:, keep] / scale[keep]
        sol = np.linalg.lstsq(a * sw[:, None], ys * (sw[:, None] if ys.ndim > 1 else sw), rcond=rcond)[0]
        coef[keep] = sol / (scale[keep][:, None] if ys.ndim > 1 else scale[keep])
    return coef, keep


@dataclass
class _CellProjector:
    """Weighted least-squares projection on ``[cell fixed effects, dense X]``,
    reusable on new rows (cell levels and slopes are stored)."""

    slopes: Array
    levels: Array
    fallback: float

    @classmethod
    def fit(cls, v: Array, x: Array, cells: npt.NDArray[np.int64], w: Array, n_cells: int) -> "_CellProjector":
        v = np.asarray(v, dtype=np.float64)
        if x.shape[1]:
            xd = x - _group_mean(x, cells, w)
            vd = v - _group_mean(v, cells, w)
            slopes, _ = _lstsq(xd, vd, w)
        else:
            slopes = np.zeros((0,) + v.shape[1:])
        u = v - x @ slopes if x.shape[1] else v
        u2 = u.reshape(len(cells), -1)
        den = np.bincount(cells, weights=w, minlength=n_cells)
        levels = np.empty((n_cells, u2.shape[1]))
        fallback = np.average(u2, axis=0, weights=w)
        for j in range(u2.shape[1]):
            num = np.bincount(cells, weights=w * u2[:, j], minlength=n_cells)
            levels[:, j] = np.where(den > 0, num / np.where(den > 0, den, 1.0), fallback[j])
        if v.ndim == 1:
            levels, fallback = levels[:, 0], float(fallback[0])
        return cls(slopes=slopes, levels=levels, fallback=fallback)

    def predict(self, x: Array, cells: npt.NDArray[np.int64]) -> Array:
        seen = cells < len(self.levels)
        lev = self.levels[np.clip(cells, 0, len(self.levels) - 1)]
        if not np.all(seen):
            lev = np.where(seen.reshape((-1,) + (1,) * (lev.ndim - 1)), lev, self.fallback)
        return lev + (x @ self.slopes if x.shape[1] else 0.0)


def _poisson_fe(
    y: Array, x: Array, w: Array, demeaner: _Demeaner, offset: Array | None = None,
    tol: float = 1e-10, max_iter: int = 200,
) -> dict[str, Any]:
    """Weighted Poisson pseudo-ML with absorbed fixed effects (IRLS, as ppmlhdfe).

    Returns slopes ``b``, fitted means ``mu`` and index ``eta`` (offset included),
    the final IRLS weights ``w * mu`` and convergence information.  The outcome is
    rescaled by its weighted mean internally (the fixed effects absorb the scale).
    """
    n = len(y)
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)
    scale = float(np.average(y, weights=w))
    ys = y / scale
    mu = 0.5 * (ys + 1.0)
    eta = np.log(mu)
    b = np.zeros(x.shape[1])
    deviance_old = np.inf
    converged = False
    for iteration in range(1, max_iter + 1):
        z = eta - offset + (ys - mu) / mu
        lam = w * mu
        lam = lam / lam.mean()
        stacked = demeaner(np.column_stack([x, z]), lam)
        xd, zd = stacked[:, :-1], stacked[:, -1]
        b, _ = _lstsq(xd, zd, lam)
        resid = zd - xd @ b
        eta_new = np.clip(offset + z - resid, -700, 700)
        step = float(np.max(np.abs(eta_new - eta)))
        eta = eta_new
        mu = np.exp(eta)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev_terms = np.where(ys > 0, ys * np.log(ys / mu), 0.0) - (ys - mu)
        deviance = 2.0 * float(np.sum(w * dev_terms))
        # IRLS converges quadratically: stop on the change in the linear index
        # (a deviance criterion alone leaves the slopes accurate only to sqrt(tol))
        if step <= 1e-10 or abs(deviance - deviance_old) <= 1e-15 * max(1.0, abs(deviance)):
            converged = True
            break
        deviance_old = deviance
    log_scale = np.log(scale)
    return {
        "b": b,
        "mu": mu * scale,
        "eta": eta + log_scale,
        "lambda": w * mu * scale,
        "converged": converged,
        "iterations": iteration,
    }


def _ols_fe(y: Array, x: Array, w: Array, demeaner: _Demeaner) -> dict[str, Any]:
    stacked = demeaner(np.column_stack([x, y]), w)
    xd, yd = stacked[:, :-1], stacked[:, -1]
    b, keep = _lstsq(xd, yd, w)
    resid = yd - xd @ b
    return {"b": b, "resid": resid, "fitted": y - resid, "keep": keep, "xd": xd}


def _union_find_components(unit: npt.NDArray[np.int64], cell: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Connected components of the bipartite (unit, cell) graph, returned per cell."""
    n_cells = int(cell.max()) + 1
    parent = np.arange(n_cells)

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    order = np.lexsort((cell, unit))
    u_sorted, c_sorted = unit[order], cell[order]
    starts = np.r_[0, np.flatnonzero(np.diff(u_sorted)) + 1]
    for s, e in zip(starts, np.r_[starts[1:], len(order)]):
        cs = np.unique(c_sorted[s:e])
        r0 = find(int(cs[0]))
        for c in cs[1:]:
            r = find(int(c))
            if r != r0:
                parent[r] = r0
    roots = np.array([find(c) for c in range(n_cells)])
    return _codes(roots)


# --------------------------------------------------------------------------- results


@dataclass
class PanelReadingBResults:
    """Results of :class:`PanelReadingB`.

    ``estimates`` and ``standard_errors`` have shape ``(n_targets, 1)`` (one interest
    variable), as in :class:`JointGeneralizedDREAMResults`.  ``influence_functions``
    holds one column per target with ``theta_hat - theta = sum_i p_i IF_i`` for
    ``p = weights / weights.sum()``; the covariance is clustered by unit unless other
    clusters are given (they should nest units).  ``plugin_*`` are the plug-in
    estimates with sieve delta-method influence functions.
    """

    targets: tuple[Target, ...]
    estimates: Array
    standard_errors: Array
    plugin_estimates: Array
    plugin_standard_errors: Array
    exog_names: list[str]
    endog_name: str
    interest_indices: list[int]
    beta: Array
    n_folds: int
    nobs: int
    n_units: int
    riesz: str
    influence_functions: Array
    plugin_influence_functions: Array
    covariance_matrix: Array
    plugin_covariance_matrix: Array
    fold_weights: Array
    unit_codes: npt.NDArray[np.int64]
    fold_results: list[dict[str, Any]]
    per_row: dict[str, Array]
    diagnostics: dict[str, Any]
    clusters: Any = None

    def __post_init__(self) -> None:
        z = float(stats.norm.ppf(0.975))
        name = self.exog_names[self.interest_indices[0]]
        records = []
        for k, (phi, psi) in enumerate(self.targets):
            est, se = float(self.estimates[k, 0]), float(self.standard_errors[k, 0])
            records.append({
                "phi": phi, "psi": psi, "variable": name, "estimate": est, "std_error": se,
                "ci_lower": est - z * se, "ci_upper": est + z * se,
                "plugin_estimate": float(self.plugin_estimates[k, 0]),
                "plugin_std_error": float(self.plugin_standard_errors[k, 0]),
            })
        self.elasticities = pd.DataFrame.from_records(records).set_index(["phi", "psi", "variable"])
        self.k_weights = np.column_stack([self.per_row[f"k_weight_{phi:g}_{psi:g}"] for phi, psi in self.targets])

    @property
    def interest_names(self) -> list[str]:
        return [self.exog_names[j] for j in self.interest_indices]

    def with_clusters(self, clusters=None, small_sample: bool = True) -> "PanelReadingBResults":
        """Copy with the covariance clustered on ``clusters`` (default: the unit).

        The correction term of the score is a within-unit sum, so the clusters must
        contain whole units; several columns give multiway clustering."""
        labels = self.unit_codes if clusters is None else clusters
        if clusters is not None:
            _warn_if_units_split(np.asarray(clusters), self.unit_codes)
        p = self.fold_weights / self.fold_weights.sum()
        cov = influence_covariance(self.influence_functions, p, labels, small_sample=small_sample)
        cov_plugin = influence_covariance(self.plugin_influence_functions, p, labels, small_sample=small_sample)
        shape = self.estimates.shape
        return replace(
            self, covariance_matrix=cov, plugin_covariance_matrix=cov_plugin, clusters=clusters,
            standard_errors=np.sqrt(np.maximum(np.diag(cov), 0.0)).reshape(shape),
            plugin_standard_errors=np.sqrt(np.maximum(np.diag(cov_plugin), 0.0)).reshape(shape),
        )

    def get_elasticity(self, var_name: str | None = None, *, phi: float, psi: float) -> tuple[float, float]:
        k = self.targets.index((float(phi), float(psi)))
        return float(self.estimates[k, 0]), float(self.standard_errors[k, 0])

    def summary(self) -> None:
        print("=" * 88)
        print("Panel reading (b) elasticity aggregates")
        print(f"Dependent variable: {self.endog_name} | Riesz: {self.riesz} | folds: {self.n_folds}")
        print(f"Observations: {self.nobs} | units: {self.n_units}")
        print("-" * 88)
        print(self.elasticities.to_string())
        print("=" * 88)


def _warn_if_units_split(clusters: npt.NDArray, unit: npt.NDArray[np.int64]) -> None:
    labels = clusters.reshape(len(unit), -1)
    for j in range(labels.shape[1]):
        frame = pd.DataFrame({"u": unit, "c": _codes(labels[:, j])})
        if (frame.groupby("u")["c"].nunique() > 1).any():
            warnings.warn(
                "Some units span several clusters. The reading (b) score has a within-unit "
                "correction term, so cluster by the unit or a coarsening of it (or add the unit "
                "as a further clustering dimension).",
                stacklevel=3,
            )
            return


# --------------------------------------------------------------------------- within models


class _SieveWithin:
    """FE-OLS (phi = 0) or FE-Poisson (phi != 0) with the polynomial sieve."""

    def __init__(self, est: "PanelReadingB", phi: float):
        self.est, self.phi = est, phi

    def fit(self, rows: npt.NDArray[np.int64]) -> "_SieveWithin":
        est, phi = self.est, self.phi
        x = est._design(rows, est.degree)
        w = est.w[rows]
        dm = _Demeaner(est.fe_all[rows])
        if phi == 0.0:
            res = _ols_fe(np.log(est.y[rows]), x, w, dm)
            self.b = res["b"]
            self.keep = res["keep"]
            self.converged = True
        else:
            res = _poisson_fe(est.y[rows] ** phi, x, w, dm)
            self.b = res["b"] / phi
            self.converged = res["converged"]
            self.keep = np.abs(res["b"]) > 0
        n_s = est._n_sterms(est.degree)
        absorbed = [j for j in range(n_s) if not self.keep[j]] if phi == 0.0 else []
        if absorbed:
            warnings.warn(
                f"{len(absorbed)} sieve terms in the interest regressor are absorbed by the fixed "
                "effects (no within-unit variation); they are dropped.", stacklevel=4)
        if not np.any(self.b[:n_s] != 0) and phi == 0.0:
            raise ValueError("The interest regressor has no within-unit variation")
        return self

    def profile(self, rows: npt.NDArray[np.int64]) -> tuple[Array, Array, Array]:
        """``ell_s`` (the within profile up to kappa-cell constants), ``eps`` and the
        full linear index ``X b`` (all regressors, no fixed effects)."""
        est = self.est
        x = est._design(rows, est.degree)
        dx = est._design_ds(rows, est.degree)
        n_s = est._n_sterms(est.degree)
        index = x @ self.b
        ell = x[:, :n_s] @ self.b[:n_s] + x[:, n_s:] @ self.b[n_s:]
        eps = dx @ self.b
        return ell, eps, index


class _ModelWithin:
    """Within fit with a pluggable nuisance class (``FESplineModelNuisance`` interface).

    The model receives ``[interest, controls, unit code, other fixed-effect codes]``
    with the fixed effects as trailing integer codes; target ``log Y`` with an
    identity link for ``phi = 0`` and ``Y^phi`` with a log link otherwise.
    """

    def __init__(self, est: "PanelReadingB", phi: float, spec: Mapping[str, Any]):
        self.est, self.phi = est, phi
        self.model_class = spec["model_class"]
        self.arch = dict(spec.get("arch_params", {}))
        self.fit_params = dict(spec.get("fit_params", {}))

    def _inputs(self, rows: npt.NDArray[np.int64], zero_codes: bool = False) -> Array:
        est = self.est
        cont = np.column_stack([est.s[rows], est.wmat[rows], est.cell_dummies[rows]])
        codes = est.fe_all[rows].astype(np.float64)
        if zero_codes:
            codes = np.zeros_like(codes)
        return np.column_stack([cont, codes])

    def fit(self, rows: npt.NDArray[np.int64]) -> "_ModelWithin":
        est, phi = self.est, self.phi
        n_cont = 1 + est.wmat.shape[1] + est.cell_dummies.shape[1]
        arch = dict(self.arch)
        arch.setdefault("non_fe_inputs", n_cont)
        arch.setdefault("categorical_cardinalities", [int(c.max()) + 1 for c in est.fe_all.T])
        if phi == 0.0:
            arch.setdefault("link", "identity")
            target = np.log(est.y[rows])
        else:
            arch.setdefault("link", "log")
            target = est.y[rows] ** phi
        types = {0: "continuous"}
        for j in range(1, n_cont):
            types[j] = "continuous"
        model = self.model_class(variable_types=types, **arch)
        fit_kwargs = dict(self.fit_params)
        fit_kwargs["sample_weight"] = est.w[rows]
        self.results = model.fit(self._inputs(rows), target, **fit_kwargs)
        self.converged = True
        return self

    def profile(self, rows: npt.NDArray[np.int64]) -> tuple[Array, Array, Array]:
        x0 = self._inputs(rows, zero_codes=True)
        pred, grad = self.results.derivative(x0, [0])
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        grad = np.asarray(grad, dtype=np.float64).reshape(len(rows), -1)[:, 0]
        if self.phi == 0.0:
            index = pred
            eps = grad
        else:
            if np.any(pred <= 0):
                raise ValueError("The within model must return positive means for phi != 0")
            index = np.log(pred) / self.phi
            eps = grad / (self.phi * pred)
        return index, eps, index


# --------------------------------------------------------------------------- estimator


class PanelReadingB:
    r"""Reading (b) aggregate elasticities ``theta_{phi,psi}(h)`` with unit fixed effects.

    Parameters
    ----------
    endog : array_like
        Strictly positive outcome ``Y`` (levels).
    exog : DataFrame or array
        Regressors, including the fixed-effect code columns named in
        ``fixed_effects``.  The interest regressor should be ``log x`` for an
        elasticity (derivatives are taken in the supplied column).
    fixed_effects : list of str or int, optional
        Fixed-effect columns of ``exog``.  ``unit`` must be one of them or an array.
        Every fixed effect other than the unit is absorbed additively in ``ell`` and
        crossed into the kappa cells.
    unit : str, int or array
        The panel unit (the scale ``C_i`` is constant within it).
    time : str, int or array, optional
        Period identifier.  Treated as a further absorbed fixed effect if it is not
        already in ``fixed_effects``; used for diagnostics only otherwise.
    interest : str or int
        The (one) continuous interest regressor.
    cells : list of str or int, optional
        Non-fixed-effect regressors treated as discrete cells (saturated in ``b_k``
        and in kappa).  The remaining non-fixed-effect regressors are controls ``W``
        handled by regression (linear in ``b_k(w)`` and in ``log kappa``).
    targets : sequence of (phi, psi)
        ``phi`` and ``psi`` in {0, 1} are supported (any real ``phi`` is accepted).
    weights : array_like, optional
        Analytic row weights.
    h, h_derivative : callable, optional
        Policy ``h`` evaluated on the non-fixed-effect regressor matrix (columns in
        ``exog`` order without the fixed effects); default ``h = 1``.
    kappa_cells : {'crossed', 'user', 'auto'}
        Cells of ``kappa``.  ``'crossed'`` (default): user cells crossed with every
        non-unit fixed effect, so additive fixed effects cancel from ``kappa e^ell``
        (time belongs in ``W`` when the outcome distribution changes over time).
        ``'user'``: user cells crossed only with fixed effects nested in units; the
        other absorbed effects stay in ``ell`` and the Riesz representer carries
        their block (for fine effects such as industry x year).  ``'auto'``:
        ``'crossed'`` unless more than 1% of rows would be in cells under 20 rows.
    """

    def __init__(
        self,
        endog: npt.ArrayLike,
        exog: npt.ArrayLike,
        *,
        unit: Any,
        interest: Any,
        fixed_effects: Sequence[Any] | None = None,
        time: Any = None,
        cells: Sequence[Any] | None = None,
        targets: Sequence[Target] = DEFAULT_TARGETS,
        weights: npt.ArrayLike | None = None,
        h: Callable[[Array], npt.ArrayLike] | None = None,
        h_derivative: Callable[[Array], npt.ArrayLike] | None = None,
        kappa_cells: str = "crossed",
        seed: int | None = None,
        **ignored: Any,
    ) -> None:
        self.seed = seed
        self.h, self.h_derivative = h, h_derivative
        self.targets = tuple((float(a), float(b)) for a, b in targets)
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be non-empty and without duplicates")

        y = np.asarray(endog, dtype=np.float64).reshape(-1)
        if y.size == 0 or not np.all(np.isfinite(y)) or np.any(y <= 0):
            raise ValueError("endog must be finite and strictly positive")
        self.y = y
        self.n = n = len(y)
        self.endog_name = endog.name if isinstance(endog, pd.Series) and endog.name is not None else "y"

        frame = exog if isinstance(exog, pd.DataFrame) else pd.DataFrame(np.asarray(exog).reshape(n, -1))
        if len(frame) != n:
            raise ValueError("exog and endog must have the same number of rows")
        frame = frame.reset_index(drop=True)
        names = [str(c) for c in frame.columns]
        frame.columns = names

        def column(key: Any) -> str:
            if isinstance(key, (int, np.integer)) and not isinstance(key, bool) and str(key) not in names:
                return names[int(key)]
            if str(key) not in names:
                raise KeyError(f"{key!r} is not a column of exog")
            return str(key)

        fe_names = [column(c) for c in (fixed_effects or [])]
        unit_values, unit_name = self._resolve(unit, frame, column)
        if unit_values is None:
            raise ValueError("unit is required")
        time_values, time_name = self._resolve(time, frame, column)
        self.unit = _codes(unit_values)
        self.n_units = int(self.unit.max()) + 1
        absorbed = [frame[c].to_numpy() for c in fe_names if c != unit_name]
        if time_values is not None and time_name not in fe_names:
            absorbed.append(time_values)
        self.time = None if time_values is None else _codes(time_values)
        self.absorbed = np.column_stack([_codes(c) for c in absorbed]) if absorbed else np.zeros((n, 0), np.int64)
        self.fe_all = np.column_stack([self.unit, self.absorbed]).astype(np.int64)

        regressors = [c for c in names if c not in fe_names and c != unit_name and c != time_name]
        self.exog_names = regressors
        self.interest_name = column(interest if not isinstance(interest, (list, tuple)) else _single(interest))
        if self.interest_name not in regressors:
            raise ValueError("interest must be a non-fixed-effect regressor")
        self.interest_index = regressors.index(self.interest_name)
        self.xmat = frame[regressors].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(self.xmat)):
            raise ValueError("exog must be finite")
        self.s = self.xmat[:, self.interest_index]
        if np.unique(self.s).size < 3:
            raise ValueError("the interest regressor must be continuous")

        cell_names = [column(c) for c in (cells or [])]
        controls = [c for c in regressors if c != self.interest_name and c not in cell_names]
        wmat = frame[controls].to_numpy(dtype=np.float64) if controls else np.zeros((n, 0))
        varying = np.ptp(wmat, axis=0) > 0 if controls else np.zeros(0, bool)
        self.control_names = [c for c, v in zip(controls, varying) if v]
        wmat = wmat[:, varying]
        self.w_center = wmat.mean(axis=0) if wmat.shape[1] else np.zeros(0)
        self.w_scale = wmat.std(axis=0) if wmat.shape[1] else np.zeros(0)
        self.wmat = (wmat - self.w_center) / np.where(self.w_scale > 0, self.w_scale, 1.0) if wmat.shape[1] else wmat

        self.cell_names = cell_names
        self.slope_cell = _combine_codes([frame[c].to_numpy() for c in cell_names], n)
        n_slope = int(self.slope_cell.max()) + 1
        self.cell_dummies = (
            np.eye(n_slope)[self.slope_cell][:, 1:] if n_slope > 1 else np.zeros((n, 0))
        )
        self.kcell = _combine_codes(
            [self.slope_cell] + [self.absorbed[:, j] for j in range(self.absorbed.shape[1])], n)
        self.n_kcells = int(self.kcell.max()) + 1
        # kappa cells: 'crossed' = user cells x every absorbed fixed effect (default);
        # 'user' = user cells x only the absorbed fixed effects nested in units (those
        # the within fit cannot see); the other absorbed effects stay in ell only.
        if kappa_cells not in ("crossed", "user", "auto"):
            raise ValueError("kappa_cells must be 'crossed', 'user' or 'auto'")
        if kappa_cells == "auto":
            # crossed unless more than 1% of rows would sit in cells under 20 rows
            sizes = np.bincount(self.kcell)
            kappa_cells = "user" if np.mean(sizes[self.kcell] < 20) > 0.01 else "crossed"
        self.kappa_cells = kappa_cells
        nested = [j for j in range(self.absorbed.shape[1])
                  if pd.DataFrame({"u": self.unit, "g": self.absorbed[:, j]}).groupby("u")["g"].nunique().max() == 1]
        self.free_fe_dims = [] if kappa_cells == "crossed" else [
            j for j in range(self.absorbed.shape[1]) if j not in nested]
        kept = [j for j in range(self.absorbed.shape[1]) if j not in self.free_fe_dims]
        self.qcell = _combine_codes([self.slope_cell] + [self.absorbed[:, j] for j in kept], n)
        self.n_qcells = int(self.qcell.max()) + 1

        self.w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
        if self.w.shape != (n,) or not np.all(np.isfinite(self.w)) or np.any(self.w <= 0):
            raise ValueError("weights must be finite, positive and one per row")

        self.s_mean = float(np.average(self.s, weights=self.w))
        self.s_sd = float(np.sqrt(np.average((self.s - self.s_mean) ** 2, weights=self.w)))
        self.st = (self.s - self.s_mean) / self.s_sd
        self.log_y_center = float(np.average(np.log(y), weights=self.w))

        # log-OLS benchmark (all fixed effects, all regressors), as GEAM's first stage
        ols = _ols_fe(np.log(y), self.xmat, self.w, _Demeaner(self.fe_all))
        self.beta = np.asarray(ols["b"], dtype=np.float64)

    @staticmethod
    def _resolve(key: Any, frame: pd.DataFrame, column: Callable[[Any], str]) -> tuple[Any, str | None]:
        if key is None:
            return None, None
        if isinstance(key, (str, int, np.integer)) and not isinstance(key, bool):
            name = column(key)
            return frame[name].to_numpy(), name
        values = np.asarray(key).reshape(-1)
        if len(values) != len(frame):
            raise ValueError("unit/time arrays must have one entry per row")
        return values, None

    # ---------------------------------------------------------------- sieve design

    def _interactors(self, rows: npt.NDArray[np.int64]) -> Array:
        if not self.interact:
            return np.zeros((len(rows), 0))
        return np.column_stack([self.cell_dummies[rows], self.wmat[rows]])

    def _n_sterms(self, degree: int) -> int:
        n_int = (self.cell_dummies.shape[1] + self.wmat.shape[1]) if self.interact else 0
        return degree * (1 + n_int)

    def _sterms(self, rows: npt.NDArray[np.int64], degree: int, derivative: bool = False) -> Array:
        st = self.st[rows]
        z = self._interactors(rows)
        blocks = []
        for k in range(1, degree + 1):
            base = (k * st ** (k - 1) / self.s_sd) if derivative else st ** k
            blocks.append(base[:, None])
            if z.shape[1]:
                blocks.append(base[:, None] * z)
        return np.column_stack(blocks)

    def _design(self, rows: npt.NDArray[np.int64], degree: int) -> Array:
        """``[s-terms, controls, user-cell dummies]`` (no constant; fixed effects absorbed)."""
        return np.column_stack([self._sterms(rows, degree), self.wmat[rows], self.cell_dummies[rows]])

    def _design_ds(self, rows: npt.NDArray[np.int64], degree: int) -> Array:
        d = self._sterms(rows, degree, derivative=True)
        return np.column_stack([d, np.zeros((len(rows), self.wmat.shape[1] + self.cell_dummies.shape[1]))])

    def _main_effects(self, rows: npt.NDArray[np.int64]) -> Array:
        return np.column_stack([self.wmat[rows], self.cell_dummies[rows]])

    def _policy(self, rows: npt.NDArray[np.int64]) -> tuple[Array, Array]:
        n = len(rows)
        if self.h is None:
            return np.ones(n), np.zeros(n)
        x = self.xmat[rows]

        def coerce(value: Any) -> Array:
            value = np.asarray(value, dtype=np.float64)
            out = np.full(n, float(value)) if value.ndim == 0 else value.reshape(-1).astype(np.float64)
            if out.shape != (n,) or not np.all(np.isfinite(out)):
                raise ValueError("h and h_derivative must return a finite scalar or one value per row")
            return out

        h = coerce(self.h(x))
        if self.h_derivative is not None:
            return h, coerce(self.h_derivative(x))
        step = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(x[:, self.interest_index]))
        xp, xm = x.copy(), x.copy()
        xp[:, self.interest_index] += step
        xm[:, self.interest_index] -= step
        return h, (coerce(self.h(xp)) - coerce(self.h(xm))) / (2 * step)

    # ---------------------------------------------------------------- diagnostics

    def connectedness(self) -> dict[str, Any]:
        """Support diagnostics: components, within-cell connectedness, stayers.

        Components are those of the bipartite (unit, kappa-cell) graph.  For each
        slope cell (the user cells; one cell without them), units with two or more
        rows in the cell and some variation in the interest regressor contribute the
        interval of their values; overlapping intervals are merged and the number of
        resulting chains is the empirical number of within-cell components.
        """
        comp = _union_find_components(self.unit, self.kcell)
        rows_per_unit = np.bincount(self.unit)
        frame = pd.DataFrame({"u": self.unit, "c": self.slope_cell, "s": self.s})
        cell_rows = []
        for c, g in frame.groupby("c"):
            stats_u = g.groupby("u")["s"].agg(["min", "max", "size"])
            movers = stats_u[(stats_u["size"] >= 2) & (stats_u["max"] > stats_u["min"])].sort_values("min")
            chains: list[list[float]] = []
            for lo, hi in zip(movers["min"], movers["max"]):
                if chains and lo <= chains[-1][1]:
                    chains[-1][1] = max(chains[-1][1], hi)
                else:
                    chains.append([lo, hi])
            s_vals = g["s"].to_numpy()
            covered = max((float(np.mean((s_vals >= lo) & (s_vals <= hi))) for lo, hi in chains), default=0.0)
            cell_rows.append({
                "slope_cell": int(c), "rows": len(g), "units": int(g["u"].nunique()),
                "movers": int(len(movers)), "within_cell_components": len(chains),
                "share_rows_in_largest_component": covered,
            })
        unit_frame = pd.DataFrame({"u": self.unit, "s": self.s}).groupby("u")["s"].agg(["min", "max", "size"])
        stayers = (unit_frame["max"] == unit_frame["min"]) & (unit_frame["size"] >= 2)
        sizes = np.bincount(self.qcell)
        return {
            "n_units": self.n_units,
            "singleton_units": int(np.sum(rows_per_unit == 1)),
            "stayer_units": int(stayers.sum()),
            "n_kappa_cells": self.n_qcells,
            "kappa_cells": self.kappa_cells,
            "smallest_kappa_cell": int(sizes.min()),
            "n_components": int(comp.max()) + 1,
            "slope_cells": cell_rows,
            "within_cell_connected": bool(all(r["within_cell_components"] <= 1 for r in cell_rows)),
        }

    # ---------------------------------------------------------------- fitting

    def fit(
        self,
        n_folds: int = 1,
        riesz: str = "automatic",
        within_model: Mapping[str, Any] | None = None,
        *,
        degree: int = 3,
        interact: bool = True,
        riesz_degree: int | None = None,
        score_degree: int = 3,
        neumann_tol: float = 1e-8,
        neumann_max_iter: int = 500,
        clusters: Any = None,
        random_state: int | None = None,
        min_kappa_cell: int = 20,
        verbose: bool = False,
    ) -> PanelReadingBResults:
        """Estimate every target.

        ``n_folds=1`` (default): no cross-fitting (Chen and Chiang 2026).
        ``n_folds>1``: nuisances are cross-fitted by unit.
        ``riesz``: ``'automatic'`` (sieve Riesz representer from the within Hessian and
        the derivative of the target in the sieve directions; default) or
        ``'explicit'`` (the Neumann-series representer with an estimated density score).
        ``within_model``: ``None`` for the polynomial sieve, or
        ``{"model_class": ..., "arch_params": {...}, "fit_params": {...}}``.
        ``degree``: sieve degree in the interest regressor; ``interact``: let the sieve
        coefficients depend on the controls and cells; ``riesz_degree``: degree of the
        Riesz basis (default ``degree``; with the sieve within fit, equal degrees make
        the 1-fold debiased estimate equal the plug-in, and its influence function the
        sieve delta-method one); ``score_degree``: polynomial degree of the
        score-matching density score used by the explicit representer.
        ``clusters``: labels for the covariance (default: the unit); they must contain
        whole units.  ``min_kappa_cell``: warn when more than 1% of rows are in kappa
        cells smaller than this.

        A pluggable ``within_model`` gets ``[interest, controls, cell dummies, unit
        code, other fixed-effect codes]`` (codes trailing, ``non_fe_inputs`` and
        ``categorical_cardinalities`` set), target ``log Y`` (identity link) or
        ``Y^phi`` (log link), and ``sample_weight``.  Its fixed-effect scalars should
        be unpenalised (e.g. ``fe_penalty=0`` for ``FENNModelNuisance``): shrinking
        the unit effects towards a common level lets the profile absorb part of
        ``C_i``, which is the confounding the within fit exists to remove.
        """
        if riesz not in ("automatic", "explicit"):
            raise ValueError("riesz must be 'automatic' or 'explicit'")
        if not isinstance(n_folds, (int, np.integer)) or n_folds < 1 or n_folds > self.n_units:
            raise ValueError("n_folds must be an integer between 1 and the number of units")
        started = time.perf_counter()
        self.degree, self.interact = int(degree), bool(interact)
        self.riesz_degree = int(riesz_degree or degree)
        self.score_degree = int(score_degree)
        self.riesz = riesz
        self.within_spec = within_model
        self.neumann_tol, self.neumann_max_iter = float(neumann_tol), int(neumann_max_iter)
        self.verbose = verbose

        all_rows = np.arange(self.n)
        if n_folds == 1:
            splits = [(all_rows, all_rows)]
        else:
            rng = np.random.default_rng(random_state if random_state is not None else self.seed)
            fold_of_unit = rng.permutation(self.n_units) % n_folds
            fold = fold_of_unit[self.unit]
            splits = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_folds)]

        phis = tuple(dict.fromkeys(phi for phi, _ in self.targets))
        per_target = {t: {key: np.zeros(self.n) for key in
                          ("omega", "heps", "corr_kappa", "corr_alpha", "corr_alpha_plugin")} for t in self.targets}
        per_phi = {phi: {key: np.zeros(self.n) for key in ("ell", "eps", "log_kappa", "rho", "lres")}
                   for phi in phis}
        fold_results: list[dict[str, Any]] = []
        diag_folds: list[dict[str, Any]] = []
        for number, (train, test) in enumerate(splits):
            fold = {"train_idx": train, "test_idx": test, "phi_results": {}, "target_results": {}}
            fold_diag: dict[str, Any] = {"fold": number}
            score = self._score_model(train) if riesz == "explicit" else None
            for phi in phis:
                nuis = self._fit_phi(phi, train)
                ev_test = self._evaluate_phi(nuis, test)
                ev_train = ev_test if n_folds == 1 else self._evaluate_phi(nuis, train)
                for key in per_phi[phi]:
                    per_phi[phi][key][test] = ev_test[key]
                fold["phi_results"][phi] = {k: ev_test[k] for k in ("ell", "eps", "log_kappa", "rho", "lres")}
                fold_diag[f"within_converged_phi{phi:g}"] = bool(nuis["within"].converged)
                for psi in [b for a, b in self.targets if a == phi]:
                    out = self._target(phi, psi, nuis, ev_train, ev_test, train, test, score, fold_diag)
                    for key in per_target[(phi, psi)]:
                        per_target[(phi, psi)][key][test] = out[key]
                    fold["target_results"][(phi, psi)] = {k: out[k] for k in ("omega", "alpha", "gamma")}
            fold_results.append(fold)
            diag_folds.append(fold_diag)

        w = self.w
        p = w / w.sum()
        est, est_plugin, infl, infl_plugin, kweights = [], [], [], [], {}
        for t in self.targets:
            a = per_target[t]
            om = a["omega"]
            sw_om = float(np.sum(w * om))
            plug = float(np.sum(w * om * a["heps"]) / sw_om)
            deb = plug + float(np.sum(w * (a["corr_kappa"] + a["corr_alpha"])) / sw_om)
            mean_om = sw_om / w.sum()
            if_deb = (om * (a["heps"] - deb) + a["corr_kappa"] + a["corr_alpha"]) / mean_om
            if_plug = (om * (a["heps"] - plug) + a["corr_kappa"] + a["corr_alpha_plugin"]) / mean_om
            if_deb -= np.sum(p * if_deb)
            if_plug -= np.sum(p * if_plug)
            est.append(deb)
            est_plugin.append(plug)
            infl.append(if_deb)
            infl_plugin.append(if_plug)
            kweights[f"k_weight_{t[0]:g}_{t[1]:g}"] = om / mean_om
        influence = np.column_stack(infl)
        influence_plugin = np.column_stack(infl_plugin)
        labels = self.unit if clusters is None else clusters
        if clusters is not None:
            _warn_if_units_split(np.asarray(clusters), self.unit)
        cov = influence_covariance(influence, p, labels)
        cov_plugin = influence_covariance(influence_plugin, p, labels)
        k = len(self.targets)

        per_row: dict[str, Array] = dict(kweights)
        for phi, arrays in per_phi.items():
            per_row[f"ell_phi{phi:g}"] = arrays["ell"]
            per_row[f"conditional_elasticity_phi{phi:g}"] = arrays["eps"]
            per_row[f"log_kappa_phi{phi:g}"] = arrays["log_kappa"]
            per_row[f"log_inner_mean_phi{phi:g}"] = arrays["log_kappa"] + arrays["ell"]
            per_row[f"within_residual_phi{phi:g}"] = arrays["rho"]
            per_row[f"kappa_residual_phi{phi:g}"] = arrays["lres"]
        for t in self.targets:
            per_row[f"correction_weight_{t[0]:g}_{t[1]:g}"] = per_target[t]["corr_alpha"]
            per_row[f"kappa_correction_{t[0]:g}_{t[1]:g}"] = per_target[t]["corr_kappa"]
        fold_id = np.full(self.n, -1)
        for number, (_, test) in enumerate(splits):
            fold_id[test] = number
        per_row["fold"] = fold_id

        # Reading (a) plug-in where it is identified without a single component:
        # psi = 0 (same as (b)) and psi = phi (outcome weights Y^phi); NaN otherwise.
        h_all, _ = self._policy(np.arange(self.n))
        reading_a = {}
        for t in self.targets:
            phi, psi = t
            eps = per_phi[phi]["eps"]
            if psi == 0.0:
                reading_a[f"{phi:g}_{psi:g}"] = float(np.sum(w * h_all * eps) / w.sum())
            elif psi == phi:
                y_w = w * np.exp(phi * (np.log(self.y) - self.log_y_center))
                reading_a[f"{phi:g}_{psi:g}"] = float(np.sum(y_w * h_all * eps) / y_w.sum())
            else:
                reading_a[f"{phi:g}_{psi:g}"] = float("nan")
        diagnostics = {
            "reading_a_plugin": reading_a,
            "connectedness": self.connectedness(),
            "folds": diag_folds,
            "settings": {
                "n_folds": int(n_folds), "riesz": riesz, "degree": self.degree, "interact": self.interact,
                "riesz_degree": self.riesz_degree, "score_degree": self.score_degree,
                "within_model": "sieve" if within_model is None else getattr(within_model["model_class"], "__name__", "custom"),
                "controls": self.control_names, "cells": self.cell_names,
                "absorbed_fixed_effects": int(self.absorbed.shape[1]),
            },
            "persistence": self._persistence(),
            "target_effective_n": {
                f"{t[0]:g}_{t[1]:g}": float((w * per_target[t]["omega"]).sum() ** 2 / np.sum((w * per_target[t]["omega"]) ** 2))
                for t in self.targets
            },
            "seconds": time.perf_counter() - started,
        }
        self.diagnostics = diagnostics
        sizes = np.bincount(self.qcell)
        small_share = float(np.mean(sizes[self.qcell] < min_kappa_cell))
        diagnostics["connectedness"]["share_rows_in_small_kappa_cells"] = small_share
        if small_share > 0.01:
            warnings.warn(
                f"{small_share:.1%} of rows are in kappa cells with fewer than {min_kappa_cell} rows "
                "(cells are the user cells crossed with every non-unit fixed effect); kappa and the "
                "psi != 0 targets are then noisy. Consider fewer absorbed fixed effects.",
                stacklevel=2,
            )
        return PanelReadingBResults(
            targets=self.targets,
            estimates=np.asarray(est).reshape(k, 1),
            standard_errors=np.sqrt(np.maximum(np.diag(cov), 0)).reshape(k, 1),
            plugin_estimates=np.asarray(est_plugin).reshape(k, 1),
            plugin_standard_errors=np.sqrt(np.maximum(np.diag(cov_plugin), 0)).reshape(k, 1),
            exog_names=list(self.exog_names),
            endog_name=self.endog_name,
            interest_indices=[self.interest_index],
            beta=np.asarray([self.beta[self.interest_index]]),
            n_folds=int(n_folds),
            nobs=self.n,
            n_units=self.n_units,
            riesz=riesz,
            influence_functions=influence,
            plugin_influence_functions=influence_plugin,
            covariance_matrix=cov,
            plugin_covariance_matrix=cov_plugin,
            fold_weights=w.copy(),
            unit_codes=self.unit,
            fold_results=fold_results,
            per_row=per_row,
            diagnostics=diagnostics,
            clusters=clusters,
        )

    # ---------------------------------------------------------------- per-phi nuisances

    def _fit_phi(self, phi: float, train: npt.NDArray[np.int64]) -> dict[str, Any]:
        within = (_SieveWithin(self, phi) if self.within_spec is None
                  else _ModelWithin(self, phi, self.within_spec)).fit(train)
        ell, _, _ = within.profile(train)
        w = self.w[train]
        z = self.wmat[train]
        cells = self.qcell[train]
        if phi == 0.0:
            kappa = _CellProjector.fit(np.log(self.y[train]) - ell, z, cells, w, self.n_qcells)
            kappa_kind = "ols"
        else:
            # Poisson GLM of (Y e^{-ell})^phi on cell fixed effects and controls:
            # log kappa^phi = level[cell] + z' pi
            r0 = np.exp(phi * (np.log(self.y[train]) - ell))
            fitp = _poisson_fe(r0, z, w, _Demeaner(cells[:, None]))
            pi = fitp["b"]
            # cell levels from the Poisson first-order condition given pi:
            # exp(level_c) = sum_c w r0 / sum_c w exp(z' pi)
            ez = np.exp(z @ pi) if z.shape[1] else np.ones(len(train))
            den = np.bincount(cells, weights=w * ez, minlength=self.n_qcells)
            num = np.bincount(cells, weights=w * r0, minlength=self.n_qcells)
            levels = np.log(np.divide(num, den, out=np.ones(self.n_qcells), where=den > 0))
            fallback = float(np.log(np.sum(w * r0) / np.sum(w * ez)))
            levels = np.where(den > 0, levels, fallback)
            kappa = _CellProjector(slopes=pi, levels=levels, fallback=fallback)
            kappa_kind = "poisson"
        return {"phi": phi, "within": within, "kappa": kappa, "kappa_kind": kappa_kind}

    def _evaluate_phi(self, nuis: Mapping[str, Any], rows: npt.NDArray[np.int64]) -> dict[str, Array]:
        """Within profile, kappa, kappa residual and within residual on ``rows``.

        The within residual re-estimates the fixed effects on ``rows`` with the
        sieve index as an offset (for held-out units the unit effect is their own;
        with one fold this reproduces the in-sample residual exactly)."""
        phi = nuis["phi"]
        within = nuis["within"]
        ell, eps, index = within.profile(rows)
        w = self.w[rows]
        z = self.wmat[rows]
        cells = self.qcell[rows]
        level = nuis["kappa"].predict(z, cells)
        log_y = np.log(self.y[rows])
        dm = _Demeaner(self.fe_all[rows])
        if phi == 0.0:
            log_kappa = level
            lres = log_y - ell - log_kappa
            r = np.ones(len(rows))
            rho = dm((log_y - index)[:, None], w)[:, 0]
            lam = w.copy()
            rest = None
        else:
            log_kappa = level / phi
            r = np.exp(phi * (log_y - ell - log_kappa))
            lres = (r - 1.0) / phi
            yphi = self.y[rows] ** phi
            pois = _poisson_fe(yphi, np.zeros((len(rows), 0)), w, dm, offset=phi * index)
            rho = yphi - pois["mu"]
            lam = pois["lambda"]
            rest = phi * index
        return {"ell": ell, "eps": eps, "log_kappa": log_kappa, "lres": lres, "r": r,
                "rho": rho, "lambda": lam, "rest": rest, "rows": rows}

    # ---------------------------------------------------------------- per-target scores

    def _target(self, phi, psi, nuis, ev_train, ev_test, train, test, score, diag) -> dict[str, Array]:
        h_tr, hp_tr = self._policy(train)
        h_te, _ = self._policy(test)
        w_tr = self.w[train]
        omega_tr = np.exp(psi * (ev_train["log_kappa"] + ev_train["ell"] - self.log_y_center))
        omega_te = np.exp(psi * (ev_test["log_kappa"] + ev_test["ell"] - self.log_y_center))
        theta_tr = float(np.sum(w_tr * omega_tr * h_tr * ev_train["eps"]) / np.sum(w_tr * omega_tr))
        a_tr = omega_tr * (h_tr * ev_train["eps"] - theta_tr)

        # gamma, the correction for estimating kappa.  The kappa GLM solves
        # sum w Z (r0 - kappa^phi) = 0 (Poisson; phi != 0) or sum w Z (log Y - ell -
        # log kappa) = 0 (OLS; phi = 0), so the exact delta-method term is
        # psi gamma log_phi(...) with gamma = k Proj_{w k}(a / k; Z), k = kappa^phi
        # (k = 1 at phi = 0).  With saturated cells this is the cell mean of a.
        z_tr, z_te = self.wmat[train], self.wmat[test]
        if psi != 0.0:
            if phi == 0.0:
                k_tr, k_te = np.ones(len(train)), np.ones(len(test))
            else:
                ref = float(np.average(phi * ev_train["log_kappa"], weights=w_tr))
                k_tr = np.exp(phi * ev_train["log_kappa"] - ref)
                k_te = np.exp(phi * ev_test["log_kappa"] - ref)
            gproj = _CellProjector.fit(a_tr / k_tr, z_tr, self.qcell[train], w_tr * k_tr, self.n_qcells)
            gamma_tr = k_tr * gproj.predict(z_tr, self.qcell[train])
            gamma_te = k_te * gproj.predict(z_te, self.qcell[test])
        else:
            gamma_tr = np.zeros(len(train))
            gamma_te = np.zeros(len(test))

        # derivative of the target in sieve directions B: F(B) = sum w [omega h dB + psi (a - gamma r) B]
        def functional(degree: int) -> tuple[Array, Array]:
            b = self._sterms(train, degree)
            db = self._sterms(train, degree, derivative=True)
            weight_b = psi * (a_tr - gamma_tr * ev_train["r"])
            f = (w_tr * omega_tr * h_tr) @ db + (w_tr * weight_b) @ b
            return f, w_tr * weight_b

        plugin_alpha = self._auto_alpha(phi, functional, ev_train, ev_test, train, test, self.degree)
        if self.riesz == "automatic":
            alpha = (plugin_alpha if self.riesz_degree == self.degree else
                     self._auto_alpha(phi, functional, ev_train, ev_test, train, test, self.riesz_degree))
        else:
            alpha = self._explicit_alpha(phi, psi, nuis, ev_train, ev_test, train, test, score,
                                         omega_tr, h_tr, hp_tr, theta_tr, gamma_tr, diag)
        w_te = self.w[test]
        return {
            "omega": omega_te,
            "heps": h_te * ev_test["eps"],
            "corr_kappa": psi * gamma_te * ev_test["lres"],
            "corr_alpha": alpha * ev_test["rho"],
            "corr_alpha_plugin": plugin_alpha * ev_test["rho"],
            "alpha": alpha,
            "gamma": gamma_te,
        }

    def _residualize_directions(self, b: Array, rows: npt.NDArray[np.int64], lam: Array) -> Array:
        """Directions ``b`` net of every fixed effect and of the controls and cell
        dummies, in the ``lam``-weighted metric of the within fit (FWL)."""
        dm = _Demeaner(self.fe_all[rows])
        main = self._main_effects(rows)
        stacked = dm(np.column_stack([b, main]), lam)
        bd, md = stacked[:, :b.shape[1]], stacked[:, b.shape[1]:]
        if md.shape[1]:
            coef, _ = _lstsq(md, bd, lam)
            bd = bd - md @ coef
        return bd

    def _auto_alpha(self, phi, functional, ev_train, ev_test, train, test, degree) -> Array:
        """Sieve (automatic) Riesz representer: ``alpha = B~ c`` with
        ``(phi_scale B~' Lambda B~) c = F(B)`` where ``B~`` are the sieve directions net
        of the nuisance directions the target is invariant to, ``Lambda = w`` (OLS) or
        ``w mu`` (Poisson) and ``phi_scale = 1`` or ``phi``.  The correction is
        ``w alpha rho``, per row, summed within units."""
        f, level_weight = functional(degree)
        lam_tr = ev_train["lambda"]
        scale = 1.0 if phi == 0.0 else phi
        if self.free_fe_dims:
            return self._auto_alpha_fe_block(f, level_weight, ev_train, ev_test, train, test, degree, scale)
        bt = self._residualize_directions(self._sterms(train, degree), train, lam_tr)
        hess = scale * (bt.T * lam_tr) @ bt
        c = np.linalg.lstsq(hess, f, rcond=1e-12)[0]
        if test is train:
            bte = bt
        else:
            bte = self._residualize_directions(self._sterms(test, degree), test, ev_test["lambda"])
        return bte @ c

    def _auto_alpha_fe_block(self, f_b, level_weight, ev_train, ev_test, train, test, degree, scale) -> Array:
        """Automatic Riesz representer when absorbed fixed effects are not kappa cells.

        The target then moves with the absorbed levels, ``F(D_g) = sum_g w psi (a -
        gamma r)``, so the representer lives in span(s-terms, absorbed dummies) net of
        the unit effects and main effects: ``alpha = M X c`` with ``scale X' Lambda M X c
        = [F(B); F(D)]``, solved by MINRES in operator form (``M``: the Lambda-weighted
        residual maker of [unit, controls, cell dummies])."""
        from scipy.sparse.linalg import LinearOperator, minres

        def pieces(rows, lam):
            unit = self.unit[rows]
            main = self._main_effects(rows)
            md = main - _group_mean(main, unit, lam) if main.shape[1] else main
            mg_inv = np.linalg.pinv((md.T * lam) @ md, rcond=1e-12) if main.shape[1] else None

            def resid(u):
                ud = u - _group_mean(u, unit, lam)
                if main.shape[1]:
                    ud = ud - md @ (mg_inv @ (md.T @ (lam * ud)))
                return ud
            return resid

        dims = [self.absorbed[:, j] for j in self.free_fe_dims]
        sizes = [int(d.max()) + 1 for d in dims]
        b_tr = self._sterms(train, degree)
        p = b_tr.shape[1]

        def x_apply(b, rows, v):
            u = b @ v[:p]
            off = p
            for d, k in zip(dims, sizes):
                u = u + v[off:off + k][d[rows]]
                off += k
            return u

        def xt_apply(b, rows, z):
            return np.concatenate([b.T @ z] + [np.bincount(d[rows], weights=z, minlength=k) for d, k in zip(dims, sizes)])

        lam = ev_train["lambda"]
        resid_tr = pieces(train, lam)
        rhs = np.concatenate([f_b] + [np.bincount(d[train], weights=level_weight, minlength=k) for d, k in zip(dims, sizes)])
        n_par = len(rhs)
        op = LinearOperator((n_par, n_par), dtype=np.float64,
                            matvec=lambda v: scale * xt_apply(b_tr, train, lam * resid_tr(x_apply(b_tr, train, v))))
        diag = scale * np.concatenate([np.einsum("ij,ij->j", b_tr * lam[:, None], b_tr)]
                                      + [np.bincount(d[train], weights=lam, minlength=k) for d, k in zip(dims, sizes)])
        precond = LinearOperator((n_par, n_par), matvec=lambda v: v / np.where(diag > 0, diag, 1.0))
        c, info = minres(op, rhs, M=precond, rtol=1e-12, maxiter=20_000)
        if info != 0:
            warnings.warn("The automatic Riesz representer with absorbed fixed effects did not converge",
                          stacklevel=4)
        if test is train:
            return resid_tr(x_apply(b_tr, train, c))
        resid_te = pieces(test, ev_test["lambda"])
        return resid_te(x_apply(self._sterms(test, degree), test, c))

    # ---------------------------------------------------------------- explicit Riesz

    def _score_model(self, train: npt.NDArray[np.int64]) -> dict[str, Any]:
        """Sieve score matching for d log f(s | cells, controls) / ds (pooled rows).

        ``score = Phi~ c`` with ``Phi`` the powers of the standardised interest
        regressor (times the interactors), ``Phi~`` net of cell fixed effects and the
        controls, and ``c = -(Phi~' W Phi~)^{-1} sum w dPhi/ds`` (Hyvarinen)."""
        deg = self.score_degree
        w = self.w[train]
        phi_b = self._sterms(train, deg)
        dphi = self._sterms(train, deg, derivative=True)
        proj = _CellProjector.fit(phi_b, self.wmat[train], self.kcell[train], w, self.n_kcells)
        phit = phi_b - proj.predict(self.wmat[train], self.kcell[train])
        c = -np.linalg.lstsq((phit.T * w) @ phit, w @ dphi, rcond=1e-12)[0]
        return {"proj": proj, "c": c, "degree": deg}

    def _score(self, score: Mapping[str, Any], rows: npt.NDArray[np.int64]) -> Array:
        phi_b = self._sterms(rows, score["degree"])
        return (phi_b - score["proj"].predict(self.wmat[rows], self.kcell[rows])) @ score["c"]

    def _explicit_alpha(self, phi, psi, nuis, ev_train, ev_test, train, test, score,
                        omega_tr, h_tr, hp_tr, theta_tr, gamma_tr, diag) -> Array:
        """Explicit representer (spec, part 2): ``alpha_t = (g_t - gbar_i) / (phi mu_t)``
        (``phi = 0``: ``g_t - gbar_i``) with ``g`` solving
        ``E[(g - gbar) | X_t, W_t] = q`` by Neumann iterations (backfitting), and
        ``q = -omega (h' + h s + psi theta) - psi gamma rbar``."""
        w = self.w[train]
        z = self.wmat[train]
        cells = self.kcell[train]
        s_hat = self._score(score, train)
        if phi != 0.0 and psi != 0.0:
            # rbar(x, w) = E[r | X, W]: sieve regression on [s-terms, controls, cells]
            xr = self._sterms(train, self.degree)
            rproj = _CellProjector.fit(ev_train["r"], np.column_stack([xr, z]), cells, w, self.n_kcells)
            rbar = rproj.predict(np.column_stack([xr, z]), cells)
        else:
            rbar = np.ones(len(train))
        q = -omega_tr * (hp_tr + h_tr * s_hat + psi * theta_tr) - psi * gamma_tr * rbar
        # E[q | W] = 0 in the population; impose it in the sample
        qc = self.qcell[train]
        q = q - _CellProjector.fit(q, z, qc, w, self.n_qcells).predict(z, qc)
        g_coef, info = self._neumann(train, q)
        diag.setdefault("neumann", {})[f"{phi:g}_{psi:g}"] = info
        g_te = self._g_eval(g_coef, test)
        gbar = _group_mean(g_te, self.unit[test], self.w[test])
        v = g_te - gbar
        if phi == 0.0:
            alpha = v
        else:
            mu_t = self._history_mean(phi, ev_train, ev_test, train, test)
            alpha = v / (phi * mu_t)
        return alpha

    def _g_basis(self, rows: npt.NDArray[np.int64]) -> Array:
        return np.column_stack([self._sterms(rows, self.riesz_degree), self.wmat[rows], self.cell_dummies[rows]])

    def _g_eval(self, coef: tuple[Array, Array], rows: npt.NDArray[np.int64]) -> Array:
        dense, levels = coef
        cells = self.kcell[rows]
        lev = np.where(cells < len(levels), levels[np.clip(cells, 0, len(levels) - 1)], 0.0)
        return self._g_basis(rows) @ dense + lev

    def _neumann(self, train: npt.NDArray[np.int64], q: Array) -> tuple[tuple[Array, Array], dict[str, Any]]:
        """Solve ``sum_t w_t (g_t - gbar_i) delta_t = sum_t w_t q_t delta_t`` for all
        ``delta`` in span(basis, kappa cells) by the Neumann series
        ``g <- P_a[w q / a] + P_a[K g]`` with ``a = w (1 - w / W_i)`` and
        ``(K g)_t = sum_{s != t} w_s g_s / (W_i - w_t)`` (backfitting regressions).
        Component constants (the kernel) are projected out at every step.  Falls back
        to conjugate gradients on the equivalent within-Gram system if the series has
        not converged after ``neumann_max_iter`` steps."""
        w = self.w[train]
        unit = self.unit[train]
        cells = self.kcell[train]
        gd = self._g_basis(train)
        wu = np.bincount(unit, weights=w, minlength=self.n_units)[unit]
        a = w * (1.0 - w / wu)
        comp = _union_find_components(unit, cells)
        comp_of_cell = np.zeros(self.n_kcells, dtype=np.int64)
        comp_of_cell[cells] = comp[cells]
        a_safe = a.copy()

        # cell-FWL solver for (G' A G) x = G' u, G = [gd, cell dummies]
        den = np.bincount(cells, weights=a, minlength=self.n_kcells)
        gd_c = gd - _group_mean(gd, cells, a_safe) if a.sum() > 0 else gd
        schur = (gd_c.T * a) @ gd_c
        schur_inv = np.linalg.pinv(schur, rcond=1e-12)

        def solve(u: Array) -> tuple[Array, Array]:
            bd = gd.T @ u
            bc = np.bincount(cells, weights=u, minlength=self.n_kcells)
            ratio = np.divide(bc, den, out=np.zeros(self.n_kcells), where=den > 0)
            # x_d = S^{-1}(b_d - Gd' A D (D'AD)^{-1} b_D)
            adj = gd.T @ (a * ratio[cells])
            xd = schur_inv @ (bd - adj)
            xc = ratio - np.divide(np.bincount(cells, weights=a * (gd @ xd), minlength=self.n_kcells), den,
                                   out=np.zeros(self.n_kcells), where=den > 0)
            return xd, xc

        def evaluate(coef: tuple[Array, Array]) -> Array:
            return gd @ coef[0] + coef[1][cells]

        def center(coef: tuple[Array, Array]) -> tuple[Array, Array]:
            g = evaluate(coef)
            k = int(comp.max()) + 1
            num = np.bincount(comp[cells], weights=a * g, minlength=k)
            dd = np.bincount(comp[cells], weights=a, minlength=k)
            shift = np.divide(num, dd, out=np.zeros(k), where=dd > 0)
            return coef[0], coef[1] - shift[comp_of_cell]

        def k_apply(g: Array) -> Array:
            sums = np.bincount(unit, weights=w * g, minlength=self.n_units)[unit]
            out = np.zeros_like(g)
            ok = wu - w > 0
            out[ok] = (sums[ok] - w[ok] * g[ok]) / (wu[ok] - w[ok])
            return out

        wq = w * q
        coef = center(solve(wq))
        g = evaluate(coef)
        norm = lambda v: float(np.sqrt(np.sum(a * v ** 2)))
        ratios, converged, it = [], False, 0
        prev_delta = None
        for it in range(1, self.neumann_max_iter + 1):
            new = center(solve(wq + a * k_apply(g)))
            g_new = evaluate(new)
            delta = norm(g_new - g)
            if prev_delta is not None and prev_delta > 0:
                ratios.append(delta / prev_delta)
            prev_delta = delta
            coef, g = new, g_new
            if delta <= self.neumann_tol * max(norm(g), 1e-300):
                converged = True
                break
        rho_star = float(np.median(ratios[-10:])) if ratios else 0.0
        method = "neumann"
        if not converged:
            coef = self._within_gram_cg(train, gd, wq, coef)
            method = "cg"
        return coef, {"iterations": it, "converged": converged, "rho_star": rho_star, "method": method}

    def _within_gram_cg(self, train, gd, wq, start) -> tuple[Array, Array]:
        """Conjugate gradients on ``G' W M_unit G c = G' w q`` (G = [gd, kappa-cell dummies])."""
        from scipy.sparse.linalg import LinearOperator, cg

        w = self.w[train]
        unit = self.unit[train]
        cells = self.kcell[train]
        p, k = gd.shape[1], self.n_kcells

        def g_of(v):
            return gd @ v[:p] + v[p:][cells]

        def gt(u):
            return np.r_[gd.T @ u, np.bincount(cells, weights=u, minlength=k)]

        def matvec(v):
            g = g_of(v)
            return gt(w * (g - _group_mean(g, unit, w)))

        op = LinearOperator((p + k, p + k), matvec=matvec, dtype=np.float64)
        diag = np.r_[np.einsum("ij,ij->j", gd * w[:, None], gd), np.bincount(cells, weights=w, minlength=k)]
        precond = LinearOperator((p + k, p + k), matvec=lambda v: v / np.where(diag > 0, diag, 1.0))
        x, info = cg(op, gt(wq), x0=np.r_[start[0], start[1]], M=precond, rtol=1e-10, maxiter=20_000)
        if info != 0:
            warnings.warn("The explicit Riesz representer did not converge (CG)", stacklevel=4)
        return x[:p], x[p:]

    def _history_mean(self, phi, ev_train, ev_test, train, test) -> Array:
        """Crude regression for ``E[Y_t^phi | X_i, W_i]``: Poisson of ``Y^phi`` on
        unit-level summaries of the history plus kappa-cell effects, with the fitted
        within index as an offset.  Only its product with the error in ``ell``
        matters, so a rough fit suffices."""

        def features(rows):
            unit = self.unit[rows]
            w = self.w[rows]
            st = self.st[rows]
            cols = [st, st ** 2, st ** 3]
            if self.wmat.shape[1]:
                cols += list(self.wmat[rows].T)
            m = _group_mean(np.column_stack(cols), unit, w)
            n_i = np.bincount(unit, minlength=self.n_units)[unit].astype(float)
            return np.column_stack([m, np.log(n_i)])

        f_tr = features(train)
        yphi = self.y[train] ** phi
        fit = _poisson_fe(yphi, f_tr, self.w[train], _Demeaner(self.kcell[train][:, None]), offset=ev_train["rest"])
        ez = np.exp(ev_train["rest"] + f_tr @ fit["b"])
        den = np.bincount(self.kcell[train], weights=self.w[train] * ez, minlength=self.n_kcells)
        num = np.bincount(self.kcell[train], weights=self.w[train] * yphi, minlength=self.n_kcells)
        levels = np.log(np.divide(num, den, out=np.ones(self.n_kcells), where=den > 0))
        fallback = float(np.log(np.sum(self.w[train] * yphi) / np.sum(self.w[train] * ez)))
        levels = np.where(den > 0, levels, fallback)
        # The unit-level factor E[C^phi | history] enters alpha as 1/mu, so a few
        # badly underpredicted units can dominate the correction: clip the factor
        # to its training 1st-99th percentiles (a rough fit suffices, see above).
        lo, hi = np.quantile(f_tr @ fit["b"], [0.01, 0.99])
        cells = self.kcell[test]
        factor_te = np.clip(features(test) @ fit["b"], lo, hi)
        return np.exp(ev_test["rest"] + factor_te + levels[cells])

    def _persistence(self) -> dict[str, float]:
        """Persistence of the interest regressor within units.

        ``within_share``: share of the variance of ``s`` net of kappa cells and
        controls that is within units.  ``rho_star``: the contraction rate of the
        Neumann iteration on the Riesz basis, i.e. the largest ``|lambda|`` of
        ``P_a K``, from ``P_a K = I - (G' A G)^{-1} G' W M_unit G`` (``M_unit``: within
        demeaning).  Directions constant within units (the kernel) are excluded.
        ``rho_star`` near 1 means near-stayers and a large variance."""
        from scipy.linalg import eigh

        rows = np.arange(self.n)
        w = self.w
        z = self.wmat
        resid = self.s - _CellProjector.fit(self.s, z, self.kcell, w, self.n_kcells).predict(z, self.kcell)
        within = resid - _group_mean(resid, self.unit, w)
        total = float(np.sum(w * resid ** 2))
        share = float(np.sum(w * within ** 2) / total) if total > 0 else float("nan")
        gd = np.column_stack([self._sterms(rows, getattr(self, "riesz_degree", 1)), z, self.cell_dummies])
        wu = np.bincount(self.unit, weights=w)[self.unit]
        a = w * (1 - w / wu)
        if a.sum() <= 0:
            return {"within_share": share, "rho_star": float("nan")}
        gd = gd - _group_mean(gd, self.kcell, a)
        gd = gd / np.maximum(np.sqrt(np.average(gd ** 2, axis=0, weights=a)), 1e-300)
        gram_a = (gd.T * a) @ gd
        gdw = gd - _group_mean(gd, self.unit, w)
        gram_w = (gdw.T * w) @ gdw
        ridge = 1e-10 * np.trace(gram_a) / max(gram_a.shape[0], 1)
        vals = eigh(gram_w, gram_a + ridge * np.eye(len(gram_a)), eigvals_only=True)
        vals = vals[vals > 1e-8 * max(float(vals.max()), 1e-300)]
        if not len(vals):
            return {"within_share": share, "rho_star": float("nan")}
        lam = 1.0 - vals
        return {"within_share": share, "rho_star": float(np.max(np.abs(lam)))}


def detect_panel_structure(
    fixed_effects: pd.DataFrame, interest: npt.ArrayLike | None = None, *, min_within_share: float = 0.01,
) -> dict[str, Any]:
    """Suggest the panel unit and period among a design's fixed-effect columns.

    Two fixed effects *cross* when every (a, b) pair occurs at most once and b
    varies within a; b is then *time-like* if it has fewer levels than a.
    Candidate units are fixed effects that (i) have a median group of at least two
    rows, (ii) keep at least ``min_within_share`` of the variance of ``interest``
    within levels (if given) and (iii) are not nested in a time-like fixed effect
    (an industry x year cell is nested in year, so it is a cell, not a unit).  The
    unit is the candidate with the most levels; the period is the time-like fixed
    effect crossing the unit with the fewest levels (``None`` if there is none).
    Every other fixed effect is absorbed.  Returns ``{"unit", "time", "absorbed",
    "within_share", "reason"}``; ``unit`` is ``None`` when no column qualifies
    (reading (b) needs a unit with within variation).
    """
    frame = fixed_effects.reset_index(drop=True)
    names = list(frame.columns)
    codes = {name: _codes(frame[name].to_numpy()) for name in names}
    levels = {name: int(c.max()) + 1 for name, c in codes.items()}

    def nested(a: str, b: str) -> bool:  # every level of a has one level of b
        return bool(pd.DataFrame({"a": codes[a], "b": codes[b]}).groupby("a")["b"].nunique().max() == 1)

    def crosses(a: str, b: str) -> bool:  # unique (a, b) pairs, b varies within a
        pairs = pd.DataFrame({"a": codes[a], "b": codes[b]})
        return (not pairs.duplicated().any()) and not nested(a, b)

    time_like = {b for a in names for b in names if a != b and levels[b] < levels[a] and crosses(a, b)}
    x = None if interest is None else np.asarray(interest, dtype=np.float64).reshape(-1)
    shares: dict[str, float] = {}
    candidates = []
    for name in names:
        sizes = np.bincount(codes[name])
        if np.median(sizes[codes[name]]) < 2:
            continue
        if x is not None:
            within = x - _group_mean(x, codes[name], np.ones(len(x)))
            total = float(np.sum((x - x.mean()) ** 2))
            shares[name] = float(np.sum(within ** 2) / total) if total > 0 else 0.0
            if shares[name] < min_within_share:
                continue
        if any(t != name and nested(name, t) for t in time_like):
            continue
        candidates.append((levels[name], name))
    if not candidates:
        return {"unit": None, "time": None, "absorbed": names, "within_share": shares,
                "reason": "no fixed effect has repeated rows with within variation in the interest "
                          "regressor outside time-like cells"}
    unit = max(candidates)[1]
    times = [(levels[t], t) for t in names if t != unit and crosses(unit, t) and levels[t] < levels[unit]]
    time_name = min(times)[1] if times else None
    absorbed = [c for c in names if c not in (unit, time_name)]
    return {"unit": unit, "time": time_name, "absorbed": absorbed, "within_share": shares,
            "reason": f"unit = {unit} (most levels among fixed effects with within variation, "
                      "not nested in a time-like effect)"}


def _single(value: Sequence[Any]) -> Any:
    if len(value) != 1:
        raise NotImplementedError("PanelReadingB supports exactly one interest regressor")
    return value[0]


ReadingB = PanelReadingB

__all__ = ["PanelReadingB", "PanelReadingBResults", "ReadingB", "DEFAULT_TARGETS", "detect_panel_structure"]
