"""
Rosenbrock (banana) function — non-convex scalar benchmark.

  f(X) = Σ_{i=1}^{d-1} [100*(X_{i+1} - X_i^2)^2 + (X_i - 1)^2]

Has a narrow curved valley along the parabola X_{i+1} = X_i^2. The global
minimum is at X* = (1,...,1) with f* = 0. The sensitivity structure is
dominated by sequential two-way interactions (X_i, X_{i+1}).

References
----------
Rosenbrock, H. H. (1960). An automatic method for finding the greatest or
  least value of a function. The Computer Journal, 3(3), 175-184.
  https://doi.org/10.1093/comjnl/3.3.175

Kucherenko, S., & Shah, N. (2007). The importance of being global. Application
  of global sensitivity analysis in Monte Carlo option pricing.
  Wilmott Magazine, 82-91.
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class Rosenbrock(BenchmarkFunction):
    """
    Rosenbrock function (d flexible, default 4).

    Domain: X_i ~ U[-2, 2].
    No analytical Sobol indices; numerically shows chain of interactions.
    """

    name = "Rosenbrock"
    output_type = "scalar"
    description = "Narrow parabolic valley; sequential X_i interactions. Non-convex, non-separable."
    reference = "Rosenbrock (1960), Comput. J. 3(3). doi:10.1093/comjnl/3.3.175"

    def __init__(self, d: int = 4):
        self.d = d
        self.bounds = [(-2.0, 2.0)] * d

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        Y = np.zeros(len(X))
        for i in range(self.d - 1):
            Y += 100.0 * (X[:, i + 1] - X[:, i] ** 2) ** 2 + (X[:, i] - 1.0) ** 2
        return Y
