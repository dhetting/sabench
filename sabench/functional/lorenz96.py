"""
Lorenz 96 model — N-input functional benchmark (chaotic dynamics).

  dX_k/dt = (X_{k+1} - X_{k-2}) X_{k-1} - X_k + F

for k = 1, ..., N with periodic boundary conditions X_0 = X_N, X_{-1} = X_{N-1}.
Here the N initial conditions X_k(0) are the inputs, and F is a fixed forcing.

Lorenz (1996) designed this as a toy model of atmospheric dynamics. With F > 5/4
the system is chaotic. Used in data assimilation and ensemble SA.

References
----------
Lorenz, E. N. (1996). Predictability: A problem partly solved. In Proceedings
  of the Seminar on Predictability, ECMWF, 1-18.

Wilks, D. S. (2005). Effects of stochastic parametrizations in the Lorenz '96
  system. Quarterly Journal of the Royal Meteorological Society, 131(606),
  389-407. https://doi.org/10.1256/qj.04.03
"""
from __future__ import annotations
import numpy as np
from sabench._base import BenchmarkFunction


class Lorenz96(BenchmarkFunction):
    """
    Lorenz 96 model with initial-condition inputs (d = N, functional output).

    Inputs: X_k(0) ~ U[F-0.5, F+0.5], k=1,...,N (initial perturbations).
    Output: X_1(t) trajectory (first variable) at n_t time steps.
    """

    name        = "Lorenz96"
    output_type = "functional"
    description = ("Chaotic atmospheric toy model; butterfly-effect sensitivity. "
                   "Inputs are initial conditions; no analytical Sobol indices.")
    reference   = ("Lorenz (1996), ECMWF Seminar on Predictability. "
                   "Wilks (2005), QJRMS 131(606). doi:10.1256/qj.04.03")

    def __init__(self, N: int = 8, F: float = 8.0, n_t: int = 100,
                 t_max: float = 10.0):
        self.N     = N
        self.F     = F
        self.n_t   = n_t
        self.t_max = t_max
        self.d     = N
        self.bounds = [(F - 0.5, F + 0.5)] * N
        self.t     = np.linspace(0.0, t_max, n_t)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, N) — initial conditions

        Returns
        -------
        Y : (n, n_t) — first variable X_1(t)
        """
        n = len(X)
        F = self.F
        dt_output = self.t_max / (self.n_t - 1)
        n_substeps = max(1, int(np.ceil(dt_output / 0.01)))
        dt = dt_output / n_substeps

        state = X.copy()   # (n, N)
        out = np.empty((n, self.n_t))
        out[:, 0] = state[:, 0]

        def rhs(y):
            # Lorenz-96 equations with periodic BC
            yp1 = np.roll(y, -1, axis=1)  # X_{k+1}
            ym1 = np.roll(y,  1, axis=1)  # X_{k-1}
            ym2 = np.roll(y,  2, axis=1)  # X_{k-2}
            return (yp1 - ym2) * ym1 - y + F

        for s in range(1, self.n_t):
            for _ in range(n_substeps):
                k1 = rhs(state)
                k2 = rhs(state + 0.5 * dt * k1)
                k3 = rhs(state + 0.5 * dt * k2)
                k4 = rhs(state + dt * k3)
                state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            out[:, s] = state[:, 0]

        return out
