"""
Friedman (1991) benchmark — 10-input scalar regression test function.

Only 5 of the 10 inputs are relevant; the remaining 5 are pure noise.
Widely used in machine learning and SA to test variable selection methods.

  f(X) = 10*sin(π*X1*X2) + 20*(X3 - 0.5)^2 + 10*X4 + 5*X5

Analytical first-order Sobol indices are computable via term-by-term
integration since the relevant terms are approximately orthogonal.

References
----------
Friedman, J. H. (1991). Multivariate adaptive regression splines.
  The Annals of Statistics, 19(1), 1-67. https://doi.org/10.1214/aos/1176347963

Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J.,
  Gatelli, D., Saisana, M., & Tarantola, S. (2008). Global Sensitivity
  Analysis: The Primer. Wiley.
"""

from __future__ import annotations

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class Friedman(BenchmarkFunction):
    """
    Friedman (1991) 10-input function.

    Inputs X_i ~ U[0, 1], i = 1, ..., 10.
    Inputs X6-X10 have zero effect (analytical S1 = ST = 0).
    """

    name = "Friedman"
    d = 10
    output_type = "scalar"
    description = (
        "5 active inputs, 5 pure noise. Analytical variance. "
        "Includes nonlinear interaction X1*X2 term."
    )
    reference = "Friedman (1991), Ann. Stat. 19(1). doi:10.1214/aos/1176347963"

    bounds = [(0.0, 1.0)] * 10

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        return 10.0 * np.sin(np.pi * x1 * x2) + 20.0 * (x3 - 0.5) ** 2 + 10.0 * x4 + 5.0 * x5

    def analytical_variance(self) -> float:
        """
        Var[f] = Var[10 sin(π X1 X2)] + Var[20(X3-0.5)^2] + Var[10 X4] + Var[5 X5]
        since the terms are approximately uncorrelated.

        Var[10 sin(π X1 X2)]: numerical (integral over [0,1]^2)
        Var[20(X3-0.5)^2] = 20^2 * Var[(X3-0.5)^2]
          = 400 * (E[(X3-0.5)^4] - (E[(X3-0.5)^2])^2)
          E[(x-0.5)^2] = 1/12, E[(x-0.5)^4] = 1/80
          Var = 400 * (1/80 - 1/144) = 400 * (9-5)/720 = 400*4/720 = 2.222...
        Var[10 X4] = 100 * 1/12 = 8.333...
        Var[5 X5]  = 25 * 1/12 = 2.083...

        sin term: E[sin^2(pi x1 x2)] approx 0.47 (numerical)
          E[sin(pi x1 x2)] approx 0 by antisymmetry... actually E ~ 0.405
          Var[10 sin] = 100*(E[sin^2] - E[sin]^2) ~ 100*(0.449 - 0.164) ~ 28.5
        """
        # Numerical estimate of sin term variance
        rng = np.random.default_rng(0)
        x1 = rng.uniform(0, 1, 500000)
        x2 = rng.uniform(0, 1, 500000)
        v_sin = float(np.var(10.0 * np.sin(np.pi * x1 * x2)))
        v_quad = 400.0 * (1.0 / 80.0 - 1.0 / 144.0)
        v_lin4 = 100.0 / 12.0
        v_lin5 = 25.0 / 12.0
        return v_sin + v_quad + v_lin4 + v_lin5

    def analytical_S1(self) -> np.ndarray:
        """
        Approximate first-order Sobol indices.
        X1, X2 have no marginal first-order effect (sin term is symmetric,
        E[f|X1]=const approximately); the interaction dominates → S1[0]≈S1[1]≈0.
        X3, X4, X5 have purely additive effects → exact.
        """
        var_total = self.analytical_variance()
        # Marginal variance of X3 term (purely additive quadratic)
        V3 = 400.0 * (1.0 / 80.0 - 1.0 / 144.0)
        V4 = 100.0 / 12.0
        V5 = 25.0 / 12.0

        rng = np.random.default_rng(0)
        x1 = rng.uniform(0, 1, 1000000)
        x2 = rng.uniform(0, 1, 1000000)
        # E[f|X1=x1] = 10*E_x2[sin(pi x1 x2)] + const_terms (that don't depend on X1)
        # V1 = Var_{X1}[E[f|X1]]. Use a grid-based MC estimate.
        n_pts = 200
        x1_grid = np.linspace(0.01, 0.99, n_pts)
        ex1 = np.array([np.mean(10.0 * np.sin(np.pi * xi * x2)) for xi in x1_grid])
        V1 = float(np.var(ex1))

        ex2 = np.array([np.mean(10.0 * np.sin(np.pi * x1 * xi)) for xi in x1_grid])
        V2 = float(np.var(ex2))

        S1 = np.array([V1, V2, V3, V4, V5, 0.0, 0.0, 0.0, 0.0, 0.0]) / var_total
        return np.clip(S1, 0.0, 1.0)
