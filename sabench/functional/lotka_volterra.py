"""
Lotka-Volterra predator-prey model — 4-input functional benchmark.

  dX/dt = α X - β X Y    (prey)
  dY/dt = δ X Y - γ Y    (predator)

Classic nonlinear ODE system with periodic orbits. Used as an SA benchmark
to test functional sensitivity methods on nonlinear oscillatory systems.

References
----------
Lotka, A. J. (1910). Contribution to the theory of periodic reactions.
  Journal of Physical Chemistry, 14(3), 271-274.

Volterra, V. (1926). Fluctuations in the abundance of a species considered
  mathematically. Nature, 118, 558-560.

Gerstner, T., & Griebel, M. (1998). Numerical integration using sparse grids.
  Numerical Algorithms, 18(3), 209-232.

Rackauckas, C., & Nie, Q. (2017). DifferentialEquations.jl -- A performant and
  feature-rich ecosystem for solving differential equations in Julia.
  Journal of Open Research Software, 5(1). https://doi.org/10.5334/jors.151
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class LotkaVolterra(BenchmarkFunction):
    """
    Lotka-Volterra predator-prey system (d=4, functional output).

    Inputs
    ------
    alpha : prey growth rate    U[0.4, 0.8]
    beta  : predation rate      U[0.01, 0.04]
    delta : predator growth     U[0.01, 0.04]
    gamma : predator death rate U[0.4, 0.8]

    Output: prey population X(t) at n_t time steps.
    """

    name = "LotkaVolterra"
    d = 4
    output_type = "functional"
    description = "Predator-prey ODE; nonlinear oscillatory dynamics. No analytical Sobol indices."
    reference = "Lotka (1910); Volterra (1926)."

    bounds = [
        (0.4, 0.8),  # alpha
        (0.01, 0.04),  # beta
        (0.01, 0.04),  # delta
        (0.4, 0.8),  # gamma
    ]

    def __init__(self, n_t: int = 100, t_max: float = 30.0, X0: float = 10.0, Y0: float = 5.0):
        self.n_t = n_t
        self.t_max = t_max
        self.X0 = X0
        self.Y0 = Y0
        self.t = np.linspace(0.0, t_max, n_t)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 4)

        Returns
        -------
        Y : (n, n_t) — prey population time series
        """
        n = len(X)
        dt = self.t_max / (self.n_t - 1)
        prey = np.empty((n, self.n_t))
        prey[:, 0] = self.X0

        alpha, beta, delta, gamma = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        xv = np.full(n, float(self.X0))
        yv = np.full(n, float(self.Y0))

        for s in range(1, self.n_t):
            # RK4 step
            k1x = alpha * xv - beta * xv * yv
            k1y = delta * xv * yv - gamma * yv

            x2 = xv + 0.5 * dt * k1x
            y2 = yv + 0.5 * dt * k1y
            k2x = alpha * x2 - beta * x2 * y2
            k2y = delta * x2 * y2 - gamma * y2

            x3 = xv + 0.5 * dt * k2x
            y3 = yv + 0.5 * dt * k2y
            k3x = alpha * x3 - beta * x3 * y3
            k3y = delta * x3 * y3 - gamma * y3

            x4 = xv + dt * k3x
            y4 = yv + dt * k3y
            k4x = alpha * x4 - beta * x4 * y4
            k4y = delta * x4 * y4 - gamma * y4

            xv = np.maximum(xv + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x), 0.0)
            yv = np.maximum(yv + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y), 0.0)
            prey[:, s] = xv

        return prey
