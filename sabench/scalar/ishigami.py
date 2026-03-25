"""
Ishigami function — non-linear, non-monotonic scalar benchmark.

  f(X) = sin(X1) + a*sin²(X2) + b*X3⁴*sin(X1)

Closed-form first-order and total-effect Sobol indices.

Reference
---------
Ishigami, T., & Homma, T. (1990). An importance quantification technique
  in uncertainty analysis for computer models. In Proceedings of ISUMA '90,
  First International Symposium on Uncertainty Modelling and Analysis.
  IEEE. https://doi.org/10.1109/ISUMA.1990.151285
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class Ishigami(BenchmarkFunction):
    """
    Ishigami (1990) function.

    Parameters
    ----------
    a : float, default 7.0
    b : float, default 0.1
    """

    name = "Ishigami"
    d = 3
    output_type = "scalar"
    description = "Non-linear, non-monotonic; closed-form S1 and ST."
    reference = "Ishigami & Homma (1990), ISUMA. https://doi.org/10.1109/ISUMA.1990.151285"

    def __init__(self, a: float = 7.0, b: float = 0.1):
        self.a = a
        self.b = b
        self.bounds = [(-np.pi, np.pi)] * 3

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 3), X_i ~ U[-π, π]

        Returns
        -------
        Y : (n,)
        """
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        return np.sin(x1) + self.a * np.sin(x2) ** 2 + self.b * x3**4 * np.sin(x1)

    def analytical_S1(self) -> np.ndarray:
        """
        Closed-form first-order Sobol indices.

        V1 = (b*π⁴/5 + b²*π⁸/50 + 1/2*(1 + b*π⁴/5)²) / Var
             simplified → V1 = 1/2*(1 + b*π⁴/5)²
        V2 = a²/8
        V3 = 0
        """
        a, b = self.a, self.b
        V1 = 0.5 * (1.0 + b * np.pi**4 / 5.0) ** 2
        V2 = a**2 / 8.0
        V3 = 0.0
        V13 = b**2 * np.pi**8 * 8.0 / 225.0
        Var = V1 + V2 + V3 + V13
        return np.array([V1, V2, V3]) / Var

    def analytical_ST(self) -> np.ndarray:
        """Closed-form total-effect Sobol indices."""
        a, b = self.a, self.b
        V1 = 0.5 * (1.0 + b * np.pi**4 / 5.0) ** 2
        V2 = a**2 / 8.0
        V3 = 0.0
        V13 = b**2 * np.pi**8 * 8.0 / 225.0
        Var = V1 + V2 + V3 + V13
        VT1 = V1 + V13
        VT2 = V2
        VT3 = V13
        return np.array([VT1, VT2, VT3]) / Var

    def analytical_variance(self) -> float:
        """Total output variance."""
        a, b = self.a, self.b
        return a**2 / 8.0 + b * np.pi**4 / 5.0 + b**2 * np.pi**8 / 18.0 + 0.5
