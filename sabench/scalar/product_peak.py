"""
Product peak (Caflisch family) — d-input scalar benchmark.

  f(X) = prod_{i=1}^{d} 1 / (c_i^2 + (X_i - w_i)^2)

A sharply peaked function in the interior of [0,1]^d. The c_i parameters
control the sharpness; smaller c_i gives a narrower peak. The w_i parameters
control the location of the peak. Analytical first-order Sobol indices can
be computed via the product structure.

References
----------
Caflisch, R. E., Morokoff, W., & Owen, A. (1997). Valuation of mortgage
  backed securities using Brownian bridges to reduce effective dimension.
  Journal of Computational Finance, 1(1), 27-46.

Kucherenko, S., Rodriguez-Fernandez, M., Pantelides, C., & Shah, N. (2009).
  Monte Carlo evaluation of derivative-based global sensitivity measures.
  Reliability Engineering & System Safety, 94(7), 1135-1148.
  https://doi.org/10.1016/j.ress.2008.05.006
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class ProductPeak(BenchmarkFunction):
    """
    Product-peak function (d flexible, default 5).

    Parameters
    ----------
    c : sharpness parameters (positive), default 1/(i+1)
    w : peak locations in [0,1], default 0.5
    """

    name = "ProductPeak"
    output_type = "scalar"
    description = (
        "Interior-peaked product function; analytical S1. "
        "Sensitivity concentrated at peak location."
    )
    reference = (
        "Caflisch, Morokoff & Owen (1997), J. Comput. Finance 1(1). "
        "Kucherenko et al. (2009), RESS 94(7)."
    )

    def __init__(self, d: int = 5, c=None, w=None):
        self.d = d
        if c is None:
            c = np.array([1.0 / (i + 1) for i in range(d)])
        if w is None:
            w = np.full(d, 0.5)
        self.c = np.asarray(c, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.bounds = [(0.0, 1.0)] * d

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        Y = np.ones(len(X))
        for i in range(self.d):
            Y *= 1.0 / (self.c[i] ** 2 + (X[:, i] - self.w[i]) ** 2)
        return Y

    def analytical_S1(self) -> np.ndarray:
        """
        For f = prod_i g_i(X_i):
          E[f] = prod_i E[g_i]
          Var[f] = prod_i E[g_i^2] - (prod_i E[g_i])^2
          V_i = Var[E[f|X_i]] = E[E[f|X_i]^2] - (E[f])^2
              = E[g_i^2] * prod_{j!=i} (E[g_j])^2 - (E[f])^2
        """
        c, w, d = self.c, self.w, self.d

        def mean_g(ci, wi):
            """E[1/(c^2 + (x-w)^2)] for x~U[0,1]."""
            a, b = 0.0 - wi, 1.0 - wi
            return (np.arctan(b / ci) - np.arctan(a / ci)) / ci

        def mean_g_sq(ci, wi):
            """E[(1/(c^2 + (x-w)^2))^2] for x~U[0,1]."""
            a, b = 0.0 - wi, 1.0 - wi
            t1 = (b / (ci**2 * (ci**2 + b**2)) + np.arctan(b / ci) / ci**3) / 2
            t2 = (a / (ci**2 * (ci**2 + a**2)) + np.arctan(a / ci) / ci**3) / 2
            return t1 - t2

        mean_g_vec = np.array([mean_g(c[i], w[i]) for i in range(d)])
        mean_g_sq_vec = np.array([mean_g_sq(c[i], w[i]) for i in range(d)])

        Ef = np.prod(mean_g_vec)
        Ef2 = np.prod(mean_g_sq_vec)
        Var = Ef2 - Ef**2
        if Var < 1e-30:
            return np.ones(d) / d

        log_mean_g = np.log(mean_g_vec + 1e-300)
        S1 = np.empty(d)
        for i in range(d):
            prod_rest_sq = np.exp(2.0 * (log_mean_g.sum() - log_mean_g[i]))
            EEfi2 = mean_g_sq_vec[i] * prod_rest_sq
            S1[i] = max(0.0, (EEfi2 - Ef**2) / Var)
        return S1
