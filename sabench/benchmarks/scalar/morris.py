"""
Morris (1991) screening function — 20-input scalar benchmark.

Designed specifically for screening: has a mixture of linear, quadratic,
and two-way interaction effects. The "important" inputs are X1-X10,
the rest are negligible but indistinguishable by first-order methods alone.

Reference
---------
Morris, M. D. (1991). Factorial sampling plans for preliminary computational
  experiments. Technometrics, 33(2), 161-174.
  https://doi.org/10.2307/1269043
"""

from __future__ import annotations

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class Morris(BenchmarkFunction):
    """
    Morris (1991) 20-input function.

    No closed-form Sobol indices, but exact Morris elementary effects
    μ* = (β_i, 2βᵢⱼ, ...) are known by construction.
    """

    name = "Morris"
    d = 20
    output_type = "scalar"
    description = (
        "Mixed linear/quadratic/interaction; 10 influential inputs. No analytical Sobol indices."
    )
    reference = "Morris (1991), Technometrics 33(2). doi:10.2307/1269043"

    bounds = [(0.0, 1.0)] * 20

    # Published parameter values
    _b0 = 0.0
    _b1 = np.array([20.0] * 10 + [0.0] * 10)
    _b2 = np.zeros((20, 20))
    _b3 = np.zeros((20, 20, 20))
    _bstar = np.array([-15.0] * 6 + [0.0] * 14)

    # Interactions among inputs 1-6 (0-indexed: 0-5)
    for _i in range(6):
        for _j in range(_i + 1, 6):
            _b2[_i, _j] = -15.0

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 20), Xᵢ ~ U[0, 1]

        Returns
        -------
        Y : (n,)
        """
        n = X.shape[0]
        # Transform: ω_i = 2*(X_i - 0.5), except for i in {3,5,7} (0-indexed)
        # where ω_i = 2*(1.1*X_i/(X_i+0.1) - 0.5)
        W = 2.0 * (X - 0.5)
        for i in [2, 4, 6]:  # 3rd, 5th, 7th inputs (0-indexed)
            W[:, i] = 2.0 * (1.1 * X[:, i] / (X[:, i] + 0.1) - 0.5)

        Y = self._b0 * np.ones(n)
        # Linear terms
        for i in range(20):
            Y += self._b1[i] * W[:, i]
        # Quadratic (starred) terms
        for i in range(6):
            Y += self._bstar[i] * W[:, i] ** 2
        # Two-way interactions (inputs 1-6)
        for i in range(6):
            for j in range(i + 1, 6):
                if self._b2[i, j] != 0:
                    Y += self._b2[i, j] * W[:, i] * W[:, j]
        return Y
