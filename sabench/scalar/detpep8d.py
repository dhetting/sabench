"""
Dette & Pepelyshev (2010) 8-dimensional curved test function.

A curved, non-polynomial function with strong nonlinearity that was
designed to be challenging for polynomial response surface methods.
Has a known sensitivity ranking: X1 > X2 > X3 ≫ X4-X8.

  f(X) = 4*(X1 - 2 + 8*X2 - 8*X2^2)^2
        + (3 - 4*X2)^2
        + 16*sqrt(X3 + 1) * (2*X3 - 1)^2
        + Σ_{i=4}^{8} (i+1) * X_i

References
----------
Dette, H., & Pepelyshev, A. (2010). Generalized Latin hypercube design for
  computer experiments. Technometrics, 52(4), 421-429.
  https://doi.org/10.1198/TECH.2010.09157

Surjanovic, S., & Bingham, D. (2013). Virtual Library of Simulation Experiments.
  https://www.sfu.ca/~ssurjano
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class DetPep8D(BenchmarkFunction):
    """
    Dette-Pepelyshev 8-dimensional function.

    X_i ~ U[0, 1], i = 1, ..., 8.
    No closed-form Sobol indices.
    """

    name = "DetPep8D"
    d = 8
    output_type = "scalar"
    description = "Curved 8-input function; X1>X2>X3≫rest. No closed-form Sobol indices."
    reference = "Dette & Pepelyshev (2010), Technometrics 52(4). doi:10.1198/TECH.2010.09157"

    bounds = [(0.0, 1.0)] * 8

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        tail = sum((i + 4) * X[:, i] for i in range(3, 8))

        t1 = 4.0 * (x1 - 2.0 + 8.0 * x2 - 8.0 * x2**2) ** 2
        t2 = (3.0 - 4.0 * x2) ** 2
        t3 = 16.0 * np.sqrt(x3 + 1.0) * (2.0 * x3 - 1.0) ** 2
        return t1 + t2 + t3 + tail
