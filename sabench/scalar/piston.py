"""
Piston simulation function — 7-input scalar benchmark.

Models the cycle time C (seconds) of a piston in a cylinder.

Reference
---------
Kenett, R., & Zacks, S. (1998). Modern Industrial Statistics. Duxbury Press.
Ben-Ari, E. N., & Steinberg, D. M. (2007). Modeling data from computer
  experiments: an empirical comparison of kriging with MARS and projection
  pursuit regression. Quality Engineering, 19(4), 327-338.
  https://doi.org/10.1080/08982110701580930
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class Piston(BenchmarkFunction):
    """
    Piston cycle-time simulation (d=7).

    Inputs
    ------
    M   : piston weight [kg]          U[30,  60]
    S   : surface area [m²]           U[0.005, 0.020]
    V0  : initial gas volume [m³]     U[0.002, 0.010]
    k   : spring coefficient [N/m]    U[1000, 5000]
    P0  : atmospheric pressure [N/m²] U[90000, 110000]
    Ta  : ambient temperature [K]     U[290, 296]
    T0  : filling temperature [K]     U[340, 360]
    """

    name = "Piston"
    d = 7
    output_type = "scalar"
    description = "Piston cycle time; no simple analytical S1/ST."
    reference = "Ben-Ari & Steinberg (2007), Quality Engineering 19(4)."

    bounds = [
        (30.0, 60.0),  # M
        (0.005, 0.020),  # S
        (0.002, 0.010),  # V0
        (1000.0, 5000.0),  # k
        (90000.0, 110000.0),  # P0
        (290.0, 296.0),  # Ta
        (340.0, 360.0),  # T0
    ]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 7)

        Returns
        -------
        C : (n,) — cycle time [s]
        """
        M, S, V0, k, P0, Ta, T0 = [X[:, i] for i in range(7)]
        A = P0 * S + 19.62 * M - k * V0 / S
        V = (S / (2.0 * k)) * (np.sqrt(A**2 + 4.0 * k * (P0 * V0 * Ta / T0)) - A)
        C = 2.0 * np.pi * np.sqrt(M / (k + S**2 * P0 * V0 * Ta / (T0 * V**2)))
        return C
