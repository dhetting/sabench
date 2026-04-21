"""
Sobol G-function — canonical multi-input scalar benchmark with analytical indices.

  G(X) = prod_{i=1}^{d} g_i(X_i),   g_i(x) = (|4x - 2| + a_i) / (1 + a_i)

The a_i parameters control input importance: a_i = 0 is maximally influential,
large a_i → negligible contribution. Closed-form first-order and total-effect
indices via the Hoeffding decomposition.

References
----------
Saltelli, A., & Sobol, I. M. (1995). About the use of rank transformation in
  sensitivity analysis of model output. Reliability Engineering & System Safety,
  50(3), 225-239. https://doi.org/10.1016/0951-8320(95)00099-2

Saltelli, A., Tarantola, S., Campolongo, F., & Ratto, M. (2004).
  Sensitivity Analysis in Practice. Wiley.

Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical
  models and their Monte Carlo estimates. Mathematics and Computers in
  Simulation, 55(1-3), 271-280. https://doi.org/10.1016/S0378-4754(00)00270-6
"""

from __future__ import annotations

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class SobolG(BenchmarkFunction):
    """
    Sobol G-function (d flexible, default 8).

    Parameters
    ----------
    a : array-like
        Importance parameters, length d.  Default [0, 1, 4.5, 9, 99, 99, 99, 99]
        from Saltelli et al. (2004).
        a_i = 0  → fully influential
        a_i → ∞ → negligible
    """

    name = "SobolG"
    output_type = "scalar"
    description = (
        "Product-form function; analytical S1 and ST. "
        "Interaction structure controlled by a_i parameters."
    )
    reference = (
        "Saltelli & Sobol (1995), Rel. Eng. Sys. Safety 50(3). doi:10.1016/0951-8320(95)00099-2"
    )

    def __init__(self, a=None):
        if a is None:
            a = [0.0, 1.0, 4.5, 9.0, 99.0, 99.0, 99.0, 99.0]
        self.a = np.asarray(a, dtype=float)
        self.d = len(self.a)
        self.bounds = [(0.0, 1.0)] * self.d

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, d), X_i ~ U[0, 1]

        Returns
        -------
        Y : (n,)
        """
        a = self.a
        Y = np.ones(len(X))
        for i in range(self.d):
            gi = (np.abs(4.0 * X[:, i] - 2.0) + a[i]) / (1.0 + a[i])
            Y *= gi
        return Y

    def _partial_variances(self) -> np.ndarray:
        """D_i = 1 / (3 * (1 + a_i)^2)."""
        return 1.0 / (3.0 * (1.0 + self.a) ** 2)

    def analytical_variance(self) -> float:
        """
        Total variance via the Efron-Stein decomposition of a product function:
          Var[G] = prod_{i=1}^{d} (1 + D_i) - 1
        """
        Di = self._partial_variances()
        return float(np.prod(1.0 + Di) - 1.0)

    def analytical_S1(self) -> np.ndarray:
        """
        First-order Sobol index via Hoeffding decomposition of product function.

        For G(X) = prod_i g_i(X_i) with E[g_i] = 1 and Var[g_i] = D_i:
          V_i = Var[E[G | X_i]] = Var[g_i(X_i) * prod_{j!=i} E[g_j]] = D_i
          Var[G] = prod_i (1 + D_i) - 1
          S_i = D_i / Var[G]

        References
        ----------
        Saltelli, A., & Sobol, I. M. (1995), RESS 50(3), eq. (2).
        """
        Di = self._partial_variances()
        Var = float(np.prod(1.0 + Di) - 1.0)
        return Di / Var

    def analytical_ST(self) -> np.ndarray:
        """
        Total-effect Sobol index:
          ST_i = (1 - S_{-i})  where S_{-i} = prod_{j!=i}(1 + D_j) - 1 / Var
        Equivalently: ST_i = D_i * prod_{j=1}^{d}(1 + D_j) / (Var * (1 + D_i))
        """
        Di = self._partial_variances()
        prod_all = np.prod(1.0 + Di)
        Var = prod_all - 1.0
        ST = np.empty(self.d)
        for i in range(self.d):
            prod_excl = prod_all / (1.0 + Di[i])
            S_excl = (prod_excl - 1.0) / Var
            ST[i] = 1.0 - S_excl
        return ST
