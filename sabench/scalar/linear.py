"""
Linear additive model — analytical first-order indices equal importance weights.

  Y = Σ aᵢ * Xᵢ,    Xᵢ ~ U[lo_i, hi_i]

S_i = aᵢ²*(hi_i-lo_i)²/12 / Var(Y)

Useful as a sanity check: estimators should recover the exact analytical
values within Monte Carlo noise.

Reference
---------
Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer.
  Wiley. https://doi.org/10.1002/9780470725184
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class LinearModel(BenchmarkFunction):
    """
    Linear additive benchmark with analytical S1 = ST.

    Parameters
    ----------
    a      : coefficient vector, length d
    bounds : list of (lo, hi); defaults to U[0,1] for all
    """

    name = "Linear"
    output_type = "scalar"
    description = "Additive linear model; analytical S1 = ST = aᵢ²σᵢ²/Var."
    reference = "Saltelli et al. (2008), Global Sensitivity Analysis."

    def __init__(self, a=None, bounds=None):
        if a is None:
            a = np.array([3.0, 2.0, 1.0, 0.5, 0.1])
        self.a = np.asarray(a, dtype=float)
        self.d = len(self.a)
        if bounds is None:
            bounds = [(0.0, 1.0)] * self.d
        self.bounds = bounds

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return X @ self.a

    def _input_variances(self) -> np.ndarray:
        return np.array([(hi - lo) ** 2 / 12.0 for lo, hi in self.bounds])

    def analytical_variance(self) -> float:
        return float(np.sum(self.a**2 * self._input_variances()))

    def analytical_S1(self) -> np.ndarray:
        Vi = self.a**2 * self._input_variances()
        return Vi / Vi.sum()

    def analytical_ST(self) -> np.ndarray:
        return self.analytical_S1()  # no interactions in additive model
