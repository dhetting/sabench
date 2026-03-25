"""
Wing weight function — 10-input scalar benchmark for aircraft wing mass.

Models the structural weight (lb) of a light aircraft wing as a function of
10 design and aerodynamic parameters. A standard SA benchmark because of its
moderate dimensionality and well-studied sensitivity structure.

  W = 0.036 * Sw^0.758 * Wfw^0.0035 * (A/cos²(Λ))^0.6 * q^0.006
      * λ^0.04 * (100*tc/cos(Λ))^(-0.3) * (Nz*Wdg)^0.49
      + Sw * Wp

References
----------
Forrester, A., Sobester, A., & Keane, A. (2008). Engineering Design via
  Surrogate Models. Wiley. https://doi.org/10.1002/9780470770801

Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J.,
  Gatelli, D., Saisana, M., & Tarantola, S. (2008). Global Sensitivity
  Analysis: The Primer. Wiley. https://doi.org/10.1002/9780470725184

Surjanovic, S., & Bingham, D. (2013). Virtual Library of Simulation Experiments.
  https://www.sfu.ca/~ssurjano
"""

from __future__ import annotations

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class WingWeight(BenchmarkFunction):
    """
    Wing weight function (d=10).

    Inputs
    ------
    Sw   : wing area [ft²]               U[150, 200]
    Wfw  : weight of fuel in the wing [lb] U[220, 300]
    A    : aspect ratio                  U[6, 10]
    Λ    : quarter-chord sweep [deg]     U[-10, 10]
    q    : dynamic pressure [lb/ft²]     U[16, 45]
    λ    : taper ratio                   U[0.5, 1]
    tc   : aerofoil thickness to chord   U[0.08, 0.18]
    Nz   : ultimate load factor          U[2.5, 6]
    Wdg  : flight design gross weight [lb] U[1700, 2500]
    Wp   : paint weight [lb/ft²]         U[0.025, 0.08]
    """

    name = "WingWeight"
    d = 10
    output_type = "scalar"
    description = "Aircraft wing weight; 4-5 inputs dominant. No closed-form Sobol indices."
    reference = (
        "Forrester, Sobester & Keane (2008), Engineering Design "
        "via Surrogate Models. doi:10.1002/9780470770801"
    )

    bounds = [
        (150.0, 200.0),  # Sw
        (220.0, 300.0),  # Wfw
        (6.0, 10.0),  # A
        (-10.0, 10.0),  # Lambda (degrees)
        (16.0, 45.0),  # q
        (0.5, 1.0),  # lambda (taper)
        (0.08, 0.18),  # tc
        (2.5, 6.0),  # Nz
        (1700.0, 2500.0),  # Wdg
        (0.025, 0.08),  # Wp
    ]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 10)

        Returns
        -------
        W : (n,) — wing weight [lb]
        """
        Sw, Wfw, A, lam_deg, q, taper, tc, Nz, Wdg, Wp = (X[:, i] for i in range(10))
        lam_rad = lam_deg * np.pi / 180.0
        cos2 = np.cos(lam_rad) ** 2

        W = (
            0.036
            * Sw**0.758
            * Wfw**0.0035
            * (A / cos2) ** 0.6
            * q**0.006
            * taper**0.04
            * (100.0 * tc / np.cos(lam_rad)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + Sw * Wp
        )
        return W
