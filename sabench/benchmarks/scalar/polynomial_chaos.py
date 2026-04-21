"""
PC-representable test function — analytical Sobol indices via polynomial chaos.

A polynomial function designed so that Sobol indices can be computed
analytically through the coefficients of a Legendre polynomial expansion.
Useful for validating PCE-based sensitivity estimators.

  f(X) = c0 + Σ_i c_i P_1(X_i) + Σ_{i<j} c_ij P_1(X_i) P_1(X_j) + Σ_i c_ii P_2(X_i)

where P_1(x) = 2x-1, P_2(x) = 6x^2 - 6x + 1 are shifted Legendre polynomials
on [0,1], and the coefficients give exact Sobol indices.

References
----------
Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions.
  Reliability Engineering & System Safety, 93(7), 964-979.
  https://doi.org/10.1016/j.ress.2007.04.002

Le Gratiet, L., Marelli, S., & Sudret, B. (2017). Metamodel-based sensitivity
  analysis: Polynomial chaos expansions and Gaussian processes.
  In Handbook of Uncertainty Quantification. Springer.
"""

from __future__ import annotations

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class PCETestFunction(BenchmarkFunction):
    """
    Legendre PCE test function (d=4) with known Sobol indices.

    Sobol indices computed analytically via PCE coefficient squares.
    """

    name = "PCETestFunction"
    d = 4
    output_type = "scalar"
    description = (
        "Legendre PCE expansion; exact S1 via coefficient squares. "
        "Interaction structure fully prescribed."
    )
    reference = "Sudret (2008), RESS 93(7). doi:10.1016/j.ress.2007.04.002"

    bounds = [(0.0, 1.0)] * 4

    # Coefficients for: f = c0 + Σ c_i P1 + Σ c_ij P1 P1 + Σ c_ii P2
    # These are chosen so S1 ~ [0.4, 0.25, 0.15, 0.05] approximately
    _c0 = 1.0
    _c1 = np.array([2.0, 1.6, 1.2, 0.7])  # linear terms (first-order)
    _c12 = 1.0  # X1*X2 interaction
    _c13 = 0.5  # X1*X3 interaction
    _c2 = np.array([0.8, 0.4, 0.2, 0.1])  # quadratic self-terms

    def _p1(self, x):
        return 2.0 * x - 1.0

    def _p2(self, x):
        return 6.0 * x**2 - 6.0 * x + 1.0

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        P1 = self._p1(X)  # (n, 4)
        P2 = self._p2(X)  # (n, 4)
        Y = (
            self._c0
            + (P1 @ self._c1)
            + self._c12 * P1[:, 0] * P1[:, 1]
            + self._c13 * P1[:, 0] * P1[:, 2]
            + (P2 @ self._c2)
        )
        return Y

    def analytical_S1(self) -> np.ndarray:
        """
        For Legendre polynomials on [0,1]:
          ||P_k||^2 = 1/(2k+1)
        So Var[c * P_k(X_i)] = c^2 * (1/(2k+1))
          P1: ||P1||^2 = 1/3   → Var = c^2/3
          P2: ||P2||^2 = 1/5   → Var = c^2/5
        First-order variances (main effects only):
          V1 = c1[0]^2/3 + c2[0]^2/5
          V2 = c1[1]^2/3 + c2[1]^2/5
          ...
        Interaction terms contribute only to higher-order ANOVA.
        Total Var = Σ_i (V_i) + c12^2/9 + c13^2/9
        """
        V = self._c1**2 / 3.0 + self._c2**2 / 5.0
        Var_total = V.sum() + self._c12**2 / 9.0 + self._c13**2 / 9.0
        S1 = np.zeros(self.d)
        S1 = V / Var_total
        return S1

    def analytical_ST(self) -> np.ndarray:
        """
        Total-effect indices include interaction terms.
        ST1 includes c12 and c13 interactions; ST2 includes c12; ST3 includes c13.
        """
        V = self._c1**2 / 3.0 + self._c2**2 / 5.0
        V12 = self._c12**2 / 9.0
        V13 = self._c13**2 / 9.0
        Var_total = V.sum() + V12 + V13
        VT = V.copy()
        VT[0] += V12 + V13
        VT[1] += V12
        VT[2] += V13
        return VT / Var_total
