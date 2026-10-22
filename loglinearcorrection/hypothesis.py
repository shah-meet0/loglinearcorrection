import numpy as np
import scipy.stats as stats


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
        centered = influences - np.sum(
            probability_weights[:, None] * influences, axis=0, keepdims=True
        )
        weighted = probability_weights[:, None] * centered
        self.influence_functions = influences
        self.probability_weights = probability_weights
        self.vcov = weighted.T @ weighted
        self.vcov = 0.5 * (self.vcov + self.vcov.T)

        coefficient_test = CoefDiffTest(
            n=self.n,
            k=self.k,
            coefs=self.coefs,
            vcov=self.vcov,
        )
        self.results = coefficient_test.results
