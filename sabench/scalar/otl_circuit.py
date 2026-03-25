"""
OTL Circuit function — 6-input scalar benchmark for mid-point voltage.

Models a transformerless push-pull circuit; used as a classic SA
test case because it has one highly influential and several negligible inputs.

Reference
---------
Ben-Ari, E. N., & Steinberg, D. M. (2007). Modeling data from computer
  experiments. Quality Engineering, 19(4), 327-338.
  https://doi.org/10.1080/08982110701580930
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class OTLCircuit(BenchmarkFunction):
    """
    OTL (output-transformer-less) circuit mid-point voltage (d=6).

    Inputs
    ------
    Rb1 : resistance b1 [kΩ]    U[50, 150]
    Rb2 : resistance b2 [kΩ]    U[25,  70]
    Rf  : feedback resistance    U[0.5,  3]
    Rc1 : collector 1 resistance U[1.2,  2.5]
    Rc2 : collector 2 resistance U[0.25, 1.2]
    beta: transistor gain        U[50,  300]
    """

    name = "OTL Circuit"
    d = 6
    output_type = "scalar"
    description = "Transformer-less push-pull circuit Vm; no analytical S1/ST."
    reference = "Ben-Ari & Steinberg (2007), Quality Engineering 19(4)."

    bounds = [
        (50.0, 150.0),  # Rb1
        (25.0, 70.0),  # Rb2
        (0.5, 3.0),  # Rf
        (1.2, 2.5),  # Rc1
        (0.25, 1.2),  # Rc2
        (50.0, 300.0),  # beta
    ]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 6)

        Returns
        -------
        Vm : (n,) — mid-point voltage [V]
        """
        Rb1, Rb2, Rf, Rc1, Rc2, beta = [X[:, i] for i in range(6)]
        Vb1 = 12.0 * Rb2 / (Rb1 + Rb2)
        term = beta * (Rc2 + 9.0) / (beta * (Rc2 + 9.0) + Rf)
        Vm = (
            (Vb1 + 0.74) * term
            + 11.35 * Rf / (beta * (Rc2 + 9.0) + Rf)
            + 0.74 * Rf * beta * (Rc2 + 9.0) / ((beta * (Rc2 + 9.0) + Rf) * Rc1)
        )
        return Vm
