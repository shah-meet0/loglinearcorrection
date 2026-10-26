from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as stats


def _cluster_codes(clusters, n: int) -> np.ndarray:
    """Integer codes, one column per clustering dimension, for ``n`` rows."""
    frame = pd.DataFrame(clusters) if np.ndim(clusters) > 1 else pd.DataFrame({0: np.asarray(clusters)})
    if frame.shape[0] != n:
        raise ValueError(f"clusters must have {n} rows; got {frame.shape[0]}.")
    if frame.isna().to_numpy().any():
        raise ValueError("clusters must not contain missing values.")
    return np.column_stack([pd.factorize(frame[c])[0] for c in frame.columns])


def influence_covariance(
    influence_functions: np.ndarray,
    probability_weights: np.ndarray,
    clusters=None,
    *,
    center: bool = False,
    small_sample: bool = True,
) -> np.ndarray:
    """Covariance of estimators with ``theta_hat - theta = sum_i p_i IF_i``.

    Without ``clusters`` this is ``(p IF).T @ (p IF)``, the row-level
    (heteroskedasticity-robust) covariance. ``clusters`` is an array of labels
    with one row per observation: one column for one-way clustering, or several
    columns for multiway clustering (Cameron, Gelbach and Miller, 2011), where
    the covariance is the inclusion-exclusion sum over every intersection of the
    clustering dimensions and negative eigenvalues are set to zero. Within each
    term the weighted influences are summed by cluster, and with
    ``small_sample`` the term is scaled by ``G / (G - 1)`` for its ``G`` clusters.
    """
    influences = np.asarray(influence_functions, dtype=float)
    if influences.ndim == 1:
        influences = influences[:, None]
    p = np.asarray(probability_weights, dtype=float).reshape(-1)
    if center:
        influences = influences - np.sum(p[:, None] * influences, axis=0, keepdims=True)
    weighted = p[:, None] * influences
    if clusters is None:
        covariance = weighted.T @ weighted
        return 0.5 * (covariance + covariance.T)

    codes = _cluster_codes(clusters, weighted.shape[0])
    dims = codes.shape[1]
    covariance = np.zeros((weighted.shape[1], weighted.shape[1]))
    for size in range(1, dims + 1):
        for subset in combinations(range(dims), size):
            _, groups = np.unique(codes[:, subset], axis=0, return_inverse=True)
            groups = groups.reshape(-1)
            n_groups = int(groups.max()) + 1
            sums = np.zeros((n_groups, weighted.shape[1]))
            np.add.at(sums, groups, weighted)
            term = sums.T @ sums
            if small_sample and n_groups > 1:
                term *= n_groups / (n_groups - 1)
            covariance += term if size % 2 == 1 else -term
    covariance = 0.5 * (covariance + covariance.T)
    if dims > 1:
        values, vectors = np.linalg.eigh(covariance)
        covariance = (vectors * np.maximum(values, 0.0)) @ vectors.T
    return covariance


class HypothesisTestResult:
    def __init__(
        self,
        statistic: float,
        p_value: float,
        df: int,
        description: str = "",
        difference: float | None = None,
        standard_error: float | None = None,
    ):
        self.statistic = float(statistic)
        self.p_value = float(p_value)
        self.df = int(df)
        self.description = description
        self.difference = None if difference is None else float(difference)
        self.standard_error = (
            None if standard_error is None else float(standard_error)
        )

    def __repr__(self):
        return (f"HypothesisTestResult(statistic={self.statistic}, "
                f"p_value={self.p_value}, df={self.df}, "
                f"description='{self.description}')")


class CoefDiffTest:
    """
    Compare blocks of coefficients of length k.

    If coefs has length 2k: run k tests for block0 - block1.
    If coefs has length 3k: run 3k tests for:
        block0 - block1, block0 - block2, and block1 - block2.

    Wald tests with χ²(1) reference.
    """

    def __init__(self, n: int, k: int, coefs: np.ndarray, vcov: np.ndarray):
        self.n = int(n)              # not used in Wald χ²(1), kept for API symmetry
        self.k = int(k)
        self.coefs = np.asarray(coefs).reshape(-1)
        self.vcov = np.asarray(vcov)

        # Basic validation
        p = self.coefs.size
        if p not in (2 * self.k, 3 * self.k):
            raise ValueError(f"coefs length {p} must be 2k or 3k with k={self.k}.")
        if self.vcov.shape != (p, p):
            raise ValueError(f"vcov must be {p}x{p}. Got {self.vcov.shape}.")

        self.num_blocks = p // self.k
        if self.num_blocks == 2:
            self.pairs = [(0, 1)]
        elif self.num_blocks == 3:
            # Now also include block1 - block2
            self.pairs = [(0, 1), (0, 2), (1, 2)]
        else:
            # Should not happen because of validation above
            raise ValueError("Unsupported number of blocks.")

        self.results = self._compute_tests()

    def _block_slice(self, b: int):
        """Return slice for block b."""
        s = b * self.k
        return slice(s, s + self.k)

    def _compute_tests(self):
        results = []
        for a, b in self.pairs:
            sa, sb = self._block_slice(a), self._block_slice(b)
            diff = self.coefs[sa] - self.coefs[sb]

            # Var(diff) = Var(a) + Var(b) - Cov(a,b) - Cov(b,a)
            Vaa = self.vcov[sa, sa]
            Vbb = self.vcov[sb, sb]
            Vab = self.vcov[sa, sb]
            Vba = self.vcov[sb, sa]
            Vdiff = Vaa + Vbb - Vab - Vba

            # Per-coefficient Wald χ²(1)
            diag = np.diag(Vdiff)
            # Numerical guard. An exactly zero variance is possible when two
            # estimators are algebraically identical; distinguish that case
            # from a materially negative covariance estimate.
            tolerance = np.finfo(float).eps * max(
                1.0, float(np.max(np.abs(np.diag(self.vcov))))
            )
            if np.any(diag < -tolerance):
                raise ValueError(
                    "Negative variance detected for at least one contrast."
                )
            positive = diag > tolerance
            stats_chi = np.empty_like(diff, dtype=float)
            stats_chi[positive] = (diff[positive] ** 2) / diag[positive]
            stats_chi[~positive] = np.where(
                np.abs(diff[~positive]) <= np.sqrt(tolerance), 0.0, np.inf
            )
            pvals = 1.0 - stats.chi2.cdf(stats_chi, df=1)

            for j in range(self.k):
                desc = f"Wald test for coef index {j}: block{a} - block{b}"
                results.append(
                    HypothesisTestResult(
                        statistic=stats_chi[j],
                        p_value=pvals[j],
                        df=1,
                        description=desc,
                        difference=diff[j],
                        standard_error=np.sqrt(max(diag[j], 0.0)),
                    )
                )
        return results


class InfluenceFunctionDiffTest:
    """Wald coefficient-difference tests from jointly observed influences.

    Parameters
    ----------
    n, k, coefs
        Have the same block interpretation as :class:`CoefDiffTest`.
    influence_functions
        Observation-level influence functions with one column per coefficient
        in ``coefs``.  Each estimator must admit the linear representation
        ``theta_hat - theta = sum_i probability_weights[i] * IF[i] + op(1)``.
    weights
        Optional non-negative sampling or frequency weights.  The covariance
        is estimated from the weighted, centered influence functions, so the
        covariance between estimators is retained in every contrast.
    clusters
        Optional cluster labels (one column per clustering dimension) for a
        cluster-robust covariance; see :func:`influence_covariance`.
    small_sample
        With ``clusters``, scale each clustering term by ``G / (G - 1)``.

    Notes
    -----
    With equal weights this computes ``IF.T @ IF / n**2``.  This is the
    influence-function counterpart of the covariance matrix supplied directly
    to :class:`CoefDiffTest` and is useful when two estimates were obtained
    from the same observations.
    """

    def __init__(
        self,
        n: int,
        k: int,
        coefs: np.ndarray,
        influence_functions: np.ndarray,
        weights: np.ndarray | None = None,
        clusters=None,
        small_sample: bool = True,
    ):
        self.n = int(n)
        self.k = int(k)
        self.coefs = np.asarray(coefs, dtype=float).reshape(-1)
        influences = np.asarray(influence_functions, dtype=float)
        if influences.ndim == 1:
            influences = influences[:, None]
        if influences.shape != (self.n, self.coefs.size):
            raise ValueError(
                "influence_functions must have shape "
                f"({self.n}, {self.coefs.size}); got {influences.shape}."
            )
        if not np.all(np.isfinite(influences)):
            raise ValueError(
                "influence_functions must contain only finite values."
            )

        if weights is None:
            weights_array = np.ones(self.n, dtype=float)
        else:
            weights_array = np.asarray(weights, dtype=float).reshape(-1)
            if weights_array.shape != (self.n,):
                raise ValueError(f"weights must have length {self.n}.")
            if (
                not np.all(np.isfinite(weights_array))
                or np.any(weights_array < 0.0)
                or weights_array.sum() <= 0.0
            ):
                raise ValueError(
                    "weights must be finite, non-negative, and have positive sum."
                )

        probability_weights = weights_array / weights_array.sum()
        self.influence_functions = influences
        self.probability_weights = probability_weights
        self.vcov = influence_covariance(
            influences, probability_weights, clusters, center=True, small_sample=small_sample
        )

        coefficient_test = CoefDiffTest(
            n=self.n,
            k=self.k,
            coefs=self.coefs,
            vcov=self.vcov,
        )
        self.results = coefficient_test.results
