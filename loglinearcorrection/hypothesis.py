import numpy as np
import scipy.stats as stats


class HypothesisTestResult:
    def __init__(self, statistic: float, p_value: float, df: int, description: str = ""):
        self.statistic = float(statistic)
        self.p_value = float(p_value)
        self.df = int(df)
        self.description = description

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
            # numerical guard
            if np.any(diag <= 0):
                raise ValueError("Nonpositive variance detected for at least one contrast.")

            stats_chi = (diff ** 2) / diag
            pvals = 1.0 - stats.chi2.cdf(stats_chi, df=1)

            for j in range(self.k):
                desc = f"Wald test for coef index {j}: block{a} - block{b}"
                results.append(
                    HypothesisTestResult(
                        statistic=stats_chi[j],
                        p_value=pvals[j],
                        df=1,
                        description=desc
                    )
                )
        return results
