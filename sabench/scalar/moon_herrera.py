"""
Moon & Herrera (2012) active subspace test function — 20-input scalar.

A smooth function with a known 1-D active subspace, useful for testing
dimension-reduction methods alongside Sobol analysis. The function depends
only on a linear combination w^T x, making the sensitivity structure
dominated by one direction.

  f(X) = exp(0.01 * w^T X)    where w = [1, 1, 0.5, 0.25, ...]

References
----------
Moon, H., Dean, A. M., & Santner, T. J. (2012). Two-stage sensitivity-based
  group screening in computer experiments. Technometrics, 54(4), 376-387.
  https://doi.org/10.1080/00401706.2012.694774

Kuo, F. Y., & Sloan, I. H. (2005). Lifting the curse of dimensionality.
  Notices of the AMS, 52(11), 1320-1329.
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class MoonHerrera(BenchmarkFunction):
    """
    Moon-Herrera active-subspace function (d=20).

    Analytical first-order indices follow from the product structure.
    For f(X) = exp(c * Σ w_i X_i), with X_i ~ U[0,1]:
      Var(f) and partial variances computed via moment generating function.
    """

    name = "MoonHerrera"
    d = 20
    output_type = "scalar"
    description = (
        "Active subspace benchmark; dominant 1-D direction. "
        "Analytical S1 via MGF of product-exponential."
    )
    reference = "Moon, Dean & Santner (2012), Technometrics 54(4). doi:10.1080/00401706.2012.694774"

    bounds = [(0.0, 1.0)] * 20

    def __init__(self, c: float = 0.01):
        self.c = c
        # Weights: geometric decay w_i = 2^(-(i-1)/2)
        self.w = np.array([2.0 ** (-0.5 * i) for i in range(self.d)])

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.c * (X @ self.w))

    def analytical_S1(self) -> np.ndarray:
        """
        For f = exp(c * Σ w_i X_i), X_i ~ U[0,1] independent:
          E[exp(c w_i X_i)] = (exp(c w_i) - 1) / (c w_i)  [MGF of uniform]
          E[f] = prod_i (exp(c w_i) - 1) / (c w_i)
          E[f^2] = prod_i (exp(2 c w_i) - 1) / (2 c w_i)
          Var[f] = E[f^2] - E[f]^2

        V_i = E[E[f|X_i]^2] - E[f]^2
          E[f|X_i = x] = exp(c w_i x) * prod_{j!=i} (exp(c w_j) - 1)/(c w_j)
          E[E[f|X_i]^2] = (exp(2 c w_i)-1)/(2 c w_i) * [prod_{j!=i}(...)]^2
        """
        c, w = self.c, self.w
        d = self.d

        def mg1(wi):
            """E[exp(c wi Xi)] for Xi ~ U[0,1]."""
            cwi = c * wi
            if abs(cwi) < 1e-12:
                return 1.0
            return (np.exp(cwi) - 1.0) / cwi

        def mg2(wi):
            """E[exp(2c wi Xi)]."""
            cwi = 2.0 * c * wi
            if abs(cwi) < 1e-12:
                return 1.0
            return (np.exp(cwi) - 1.0) / cwi

        m1 = np.array([mg1(w[i]) for i in range(d)])
        m2 = np.array([mg2(w[i]) for i in range(d)])

        Ef = np.prod(m1)
        Ef2 = np.prod(m2)
        Var = Ef2 - Ef**2
        if Var < 1e-30:
            return np.ones(d) / d

        S1 = np.empty(d)
        log_m1 = np.log(m1 + 1e-300)
        total_log_m1 = log_m1.sum()
        for i in range(d):
            prod_rest_sq = np.exp(2.0 * (total_log_m1 - log_m1[i]))
            E_Efi2 = m2[i] * prod_rest_sq
            Vi = E_Efi2 - Ef**2
            S1[i] = max(0.0, Vi / Var)
        return S1
