"""
Boussinesq recession model — 3-input functional benchmark.

Models the time-varying groundwater outflow from an unconfined aquifer
following a rainfall event, based on the Boussinesq (1877) equation
linearised for small perturbations:

  Q(t) = Q0 * exp(-t / T_r)    where  T_r = L^2 * f / (3 * K * D0)

This is the hydraulic recession curve model, fundamental in catchment
hydrology and groundwater baseflow analysis.

References
----------
Boussinesq, J. (1877). Essai sur la théorie des eaux courantes. Mémoires
  présentés par divers savants à l'Académie des Sciences, 23(1), 1-680.

Brutsaert, W., & Nieber, J. L. (1977). Regionalized drought flow hydrographs
  from a mature glaciated plateau. Water Resources Research, 13(3), 637-643.
  https://doi.org/10.1029/WR013i003p00637

Tallaksen, L. M. (1995). A review of baseflow recession analysis. Journal of
  Hydrology, 165(1-4), 349-370. https://doi.org/10.1016/0022-1694(94)02540-R
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class BoussinesqRecession(BenchmarkFunction):
    """
    Boussinesq hydraulic recession model (d=3, functional output).

    Inputs
    ------
    Q0 : initial flow [m³/s]         U[1, 10]
    T_r: recession time constant [d]  U[5, 50]   (K*D0/L^2 proxy)
    k  : nonlinearity exponent        U[1, 2]    (1=linear, 2=quadratic)

    Output: Q(t) at n_t time steps.
    """

    name = "BoussinesqRecession"
    d = 3
    output_type = "functional"
    description = (
        "Groundwater recession; exponential/power-law decay. S1 analytically tractable for k=1."
    )
    reference = (
        "Boussinesq (1877); Brutsaert & Nieber (1977), WRR 13(3). doi:10.1029/WR013i003p00637"
    )

    bounds = [
        (1.0, 10.0),  # Q0
        (5.0, 50.0),  # T_r
        (1.0, 2.0),  # k
    ]

    def __init__(self, n_t: int = 100, t_max: float = 100.0):
        self.n_t = n_t
        self.t_max = t_max
        self.t = np.linspace(0.0, t_max, n_t)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 3)

        Returns
        -------
        Q : (n, n_t)
        """
        Q0 = X[:, 0, None]  # (n, 1)
        Tr = X[:, 1, None]
        k = X[:, 2, None]
        t = self.t[None, :]  # (1, n_t)

        # Generalised Boussinesq: Q(t) = Q0 * (1 + (k-1)*t/Tr)^(-1/(k-1)) for k!=1
        # For k=1: Q(t) = Q0 * exp(-t/Tr)
        # Use continuous parameterisation:
        arg = 1.0 + (k - 1.0) * t / Tr
        Q = Q0 * np.maximum(arg, 1e-10) ** (-1.0 / np.maximum(k - 1.0, 1e-6))
        # Patch k~1 with exponential:
        mask = np.abs(k - 1.0) < 0.01
        Q_exp = Q0 * np.exp(-t / Tr)
        Q = np.where(np.broadcast_to(mask, Q.shape), Q_exp, Q)
        return Q
