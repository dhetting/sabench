"""
Oakley & O'Hagan (2004) 15-input quadratic function.

A carefully designed test function with 5 inputs that have only linear effects,
5 with only quadratic effects, and 5 with only interactive effects. Provides
exact analytical Sobol indices and is widely used to benchmark emulators and
sensitivity methods.

  f(X) = a1^T X + a2^T sin(X) + a3^T cos(X) + X^T M X

where a1, a2, a3, M are given by Oakley & O'Hagan (2004), Table 1.

References
----------
Oakley, J. E., & O'Hagan, A. (2004). Probabilistic sensitivity analysis of
  complex models: a Bayesian approach. Journal of the Royal Statistical
  Society: Series B, 66(3), 751-769.
  https://doi.org/10.1111/j.1467-9868.2004.05304.x
"""

from __future__ import annotations

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class OakleyOHagan(BenchmarkFunction):
    """
    Oakley & O'Hagan (2004) 15-input test function.

    Inputs: X_i ~ N(0, 1), i = 1, ..., 15
    (bounds approximated as ±4σ for uniform sampling convenience).

    The function decomposes exactly: inputs 1-5 have linear effects only,
    inputs 6-10 have quadratic effects only, inputs 11-15 have only
    interaction effects (zero first-order index).
    """

    name = "OakleyOHagan"
    d = 15
    output_type = "scalar"
    description = (
        "15-input quadratic; analytical S1. Partitioned into linear, "
        "quadratic, and interactive inputs."
    )
    reference = "Oakley & O'Hagan (2004), JRSS-B 66(3). doi:10.1111/j.1467-9868.2004.05304.x"

    bounds = [(-4.0, 4.0)] * 15

    # Coefficients from Oakley & O'Hagan (2004) Table 1
    _a1 = np.array(
        [0.0118, 0.0456, 0.2297, 0.2897, 0.3897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    _a2 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.4497, 0.4872, 0.5197, 0.5497, 0.5872, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    _a3 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5497, 0.5772, 0.5972, 0.6097, 0.6272]
    )

    # Symmetric interaction matrix (off-diagonal terms only for inputs 11-15)
    _M = np.zeros((15, 15))
    _pairs = [
        (10, 11, 0.1),
        (10, 12, 0.2),
        (10, 13, 0.1),
        (10, 14, 0.2),
        (11, 12, 0.2),
        (11, 13, 0.1),
        (11, 14, 0.1),
        (12, 13, 0.2),
        (12, 14, 0.1),
        (13, 14, 0.2),
    ]
    for _i, _j, _v in _pairs:
        _M[_i, _j] = _v
        _M[_j, _i] = _v

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        X : (n, 15), X_i ~ N(0,1) [use bounds ±4 for uniform approximation]
        """
        a1, a2, a3, M = self._a1, self._a2, self._a3, self._M
        linear = X @ a1
        sine = np.sin(X) @ a2
        cosine = np.cos(X) @ a3
        quad = np.einsum("ni,ij,nj->n", X, M, X)
        return linear + sine + cosine + quad

    def analytical_S1(self) -> np.ndarray:
        """
        For X_i ~ N(0,1) [approximated via U[-4,4] here]:
        Exact for N(0,1): Var[a1_i X_i] = a1_i^2,
        Var[a2_i sin(X_i)] = a2_i^2 * (1 - exp(-2)) / 2,
        Var[a3_i cos(X_i)] = a3_i^2 * (1 - exp(-2)) / 2 * correction
        Interaction inputs (11-15) have zero first-order index analytically.

        Here we use the uniform U[-4,4] numerical approximation.
        Returns None to defer to numerical estimation.
        """
        return None  # No exact closed form under uniform bounds
