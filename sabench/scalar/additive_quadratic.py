"""
Purely additive quadratic model — analytical first-order and total-effect indices.

  f(X) = Σ_i (a_i * X_i^2 + b_i * X_i)

For X_i ~ U[0,1]:
  E[f] = Σ_i (a_i/3 + b_i/2)
  Var_i = a_i^2*(E[X^4]-E[X^2]^2) + b_i^2*(1/12) + 2*a_i*b_i*(E[X^3]-E[X^2]*E[X])
  (Cross terms are zero under independence)

This function has S1 = ST (no interactions), making it a clean test for
estimator variance without interaction confounding.

Reference
---------
Saltelli, A., et al. (2008). Global Sensitivity Analysis: The Primer. Wiley.
"""

from __future__ import annotations

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class AdditiveQuadratic(BenchmarkFunction):
    """
    Additive quadratic: f = Σ (a_i X_i^2 + b_i X_i), X_i ~ U[0,1].

    Analytical S1 = ST (no interactions by construction).

    Parameters
    ----------
    a, b : coefficient vectors of length d.
    """

    name = "AdditiveQuadratic"
    output_type = "scalar"
    description = "Purely additive quadratic; S1=ST exact. Convex for a_i>0, concave for a_i<0."
    reference = "Saltelli et al. (2008), Global Sensitivity Analysis. Wiley."

    def __init__(self, d: int = 5, a=None, b=None):
        self.d = d
        if a is None:
            a = np.linspace(2.0, 0.2, d)
        if b is None:
            b = np.ones(d)
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.bounds = [(0.0, 1.0)] * d

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return (self.a * X**2 + self.b * X).sum(axis=1)

    def analytical_variance(self) -> float:
        """
        Var[f] = Σ_i Var[a_i X_i^2 + b_i X_i]
        For X~U[0,1]:
          E[X] = 1/2, E[X^2] = 1/3, E[X^3] = 1/4, E[X^4] = 1/5
          Var[a X^2 + b X] = a^2*(1/5 - 1/9) + b^2/12 + 2ab*(1/4 - 1/6)
                           = a^2*4/45 + b^2/12 + ab/6
        """
        a, b = self.a, self.b
        return float(np.sum(a**2 * 4.0 / 45.0 + b**2 / 12.0 + a * b / 6.0))

    def _component_variances(self) -> np.ndarray:
        a, b = self.a, self.b
        return a**2 * 4.0 / 45.0 + b**2 / 12.0 + a * b / 6.0

    def analytical_S1(self) -> np.ndarray:
        Vi = self._component_variances()
        return Vi / Vi.sum()

    def analytical_ST(self) -> np.ndarray:
        return self.analytical_S1()
