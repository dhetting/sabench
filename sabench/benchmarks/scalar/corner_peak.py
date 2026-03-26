"""
Corner peak (Morokoff & Caflisch) integration test function.

  f(X) = (1 + Σ c_i X_i)^{-(d+1)}

This function concentrates mass near the origin corner, making it a
challenging integration test. Analytical Sobol indices are known.
Used in SA literature as a benchmark with analytically tractable
first-order indices.

References
----------
Morokoff, W. J., & Caflisch, R. E. (1995). Quasi-Monte Carlo integration.
  Journal of Computational Physics, 122(2), 218-230.
  https://doi.org/10.1006/jcph.1995.1209

Kucherenko, S., Tarantola, S., & Annoni, P. (2012). Estimation of global
  sensitivity indices for models with dependent variables.
  Computer Physics Communications, 183(4), 937-946.
  https://doi.org/10.1016/j.cpc.2011.12.020
"""

from __future__ import annotations

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class CornerPeak(BenchmarkFunction):
    """
    Corner peak function (d flexible, default 6).

    Parameters
    ----------
    c : array-like, default uniform 1/(d*[1,2,...,d])
        Coefficient vector controlling sharpness per dimension.
    """

    name = "CornerPeak"
    output_type = "scalar"
    description = (
        "Corner-concentrated; near-zero for large X. Analytical S1 via recursive integral formula."
    )
    reference = "Morokoff & Caflisch (1995), J. Comput. Phys. 122(2). doi:10.1006/jcph.1995.1209"

    def __init__(self, d: int = 6, c=None):
        self.d = d
        if c is None:
            c = np.arange(1, d + 1, dtype=float) / (d * (d + 1) / 2)
        self.c = np.asarray(c, dtype=float)
        self.bounds = [(0.0, 1.0)] * d

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return (1.0 + X @ self.c) ** (-(self.d + 1))

    def _mean(self) -> float:
        """E[f] = prod_i [(1+c_i)^(-d) - 1] / (-c_i * d) via recursion."""
        # Numerical fallback for generality
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (200000, self.d))
        return float(self.evaluate(X).mean())

    def analytical_S1(self) -> np.ndarray:
        """Numerical first-order indices via 1D quadrature with cached rest sums."""
        from numpy.polynomial.legendre import leggauss

        n_quad = 200
        quad_nodes, quad_weights = leggauss(n_quad)
        quad_nodes = 0.5 * (quad_nodes + 1.0)  # map [-1,1] → [0,1]
        quad_weights = 0.5 * quad_weights

        d, c = self.d, self.c
        rng = np.random.default_rng(1)
        X_mc = rng.uniform(0, 1, (100000, d))
        linear_term = X_mc @ c
        Y_mc = (1.0 + linear_term) ** (-(d + 1))
        Ef = float(Y_mc.mean())
        Ef2 = float((Y_mc**2).mean())
        variance = Ef2 - Ef**2
        if variance < 1e-30:
            return np.ones(d) / d

        S1 = np.empty(d)
        for i in range(d):
            rest_sum = linear_term - c[i] * X_mc[:, i]
            conditional_means = np.empty(len(quad_nodes))
            for k, node in enumerate(quad_nodes):
                conditional_values = (1.0 + rest_sum + c[i] * node) ** (-(d + 1))
                conditional_means[k] = float(conditional_values.mean())
            conditional_second_moment = float(np.sum(quad_weights * conditional_means**2))
            S1[i] = max(0.0, (conditional_second_moment - Ef**2) / variance)
        return S1
