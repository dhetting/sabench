"""
1D heat diffusion — 4-input functional/spatial benchmark.

Solves the transient heat equation on [0, L] with Dirichlet boundary conditions:

  ∂T/∂t = α ∂²T/∂x²,  T(0,t) = T_left, T(L,t) = T_right, T(x,0) = T0

Analytical solution via Fourier series. This is a spatiotemporal benchmark
(output is a 1D spatial field at time t_obs), enabling tests of both spatial
and temporal transforms.

References
----------
Carslaw, H. S., & Jaeger, J. C. (1959). Conduction of Heat in Solids (2nd ed.).
  Oxford University Press.

Iooss, B., & Saltelli, A. (2017). Introduction to sensitivity analysis.
  In Handbook of Uncertainty Quantification. Springer.
  https://doi.org/10.1007/978-3-319-12385-1_31
"""
from __future__ import annotations
import numpy as np
from sabench._base import BenchmarkFunction


class HeatDiffusion1D(BenchmarkFunction):
    """
    1D transient heat diffusion (d=4, spatial output).

    Inputs
    ------
    alpha  : thermal diffusivity [m²/s]   U[1e-7, 1e-5]
    T0     : initial temperature [°C]     U[0, 100]
    T_left : left boundary [°C]           U[0, 50]
    T_right: right boundary [°C]          U[50, 150]

    Output: temperature profile T(x, t_obs) at n_x points.
    """

    name        = "HeatDiffusion1D"
    d           = 4
    output_type = "functional"
    description = ("1D Fourier heat equation; analytical Fourier-series solution. "
                   "Spatial temperature profile output.")
    reference   = ("Carslaw & Jaeger (1959), Conduction of Heat in Solids.")

    bounds = [
        (1e-7,  1e-5),   # alpha
        (0.0,   100.0),  # T0
        (0.0,   50.0),   # T_left
        (50.0,  150.0),  # T_right
    ]

    def __init__(self, n_x: int = 64, L: float = 1.0, t_obs: float = 1000.0,
                 n_terms: int = 50):
        self.n_x    = n_x
        self.L      = L
        self.t_obs  = t_obs
        self.n_terms = n_terms
        self.x      = np.linspace(0.0, L, n_x)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 4)

        Returns
        -------
        T : (n, n_x) — temperature at t_obs
        """
        alpha, T0, T_left, T_right = [X[:, i] for i in range(4)]
        L, t = self.L, self.t_obs
        x = self.x[None, :]    # (1, n_x)

        # Steady-state solution
        T_ss = (T_left[:, None] + (T_right - T_left)[:, None] * x / L)

        # Transient correction: Fourier series sum
        T0_bc = T0[:, None]
        T_trans = np.zeros((len(X), self.n_x))
        for n in range(1, self.n_terms + 1):
            bn = (2.0 / L) * np.trapezoid(
                (T0_bc - T_ss) * np.sin(n * np.pi * x / L),
                x, axis=1
            )  # (n_samples,)
            decay = np.exp(-alpha * (n * np.pi / L)**2 * t)  # (n_samples,)
            T_trans += (bn * decay)[:, None] * np.sin(n * np.pi * x / L)

        return T_ss + T_trans
