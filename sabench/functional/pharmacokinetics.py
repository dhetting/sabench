"""
Two-compartment pharmacokinetic model — 5-input functional benchmark.

Models drug concentration in the central compartment following bolus injection:

  C(t) = A * exp(-α t) + B * exp(-β t)

where A, B, α, β are functions of the micro-rate constants and dose D.

The bi-exponential shape is analytically tractable. Sobol indices for
AUC (area under the curve) can be computed analytically.

References
----------
Gibaldi, M., & Perrier, D. (1982). Pharmacokinetics (2nd ed.).
  Marcel Dekker.

Minto, C. F., Schnider, T. W., Short, T. G., Gregg, K. M., Gentilini, A.,
  & Shafer, S. L. (2000). Response surface model for anesthetic drug
  interactions. Anesthesiology, 92(6), 1603-1616.

Lobo, E. D., & Bhatt, D. L. (2010). Pharmacokinetics and pharmacodynamics
  of cardiovascular drugs. In Cardiovascular Pharmacology. Academic Press.
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class TwoCompartmentPK(BenchmarkFunction):
    """
    Two-compartment PK model (d=5, functional output).

    Inputs
    ------
    D    : dose [mg]                  U[50, 200]
    k10  : elimination rate [1/h]     U[0.05, 0.3]
    k12  : transfer to peripheral [1/h] U[0.01, 0.2]
    k21  : return from peripheral     U[0.01, 0.15]
    V1   : central volume [L]         U[5, 20]

    Output: central compartment concentration C(t) [mg/L] at n_t times.
    """

    name = "TwoCompartmentPK"
    d = 5
    output_type = "functional"
    description = (
        "Two-compartment pharmacokinetics; bi-exponential C(t). Analytical AUC indices available."
    )
    reference = "Gibaldi & Perrier (1982), Pharmacokinetics. Marcel Dekker."

    bounds = [
        (50.0, 200.0),  # D
        (0.05, 0.30),  # k10
        (0.01, 0.20),  # k12
        (0.01, 0.15),  # k21
        (5.0, 20.0),  # V1
    ]

    def __init__(self, n_t: int = 100, t_max: float = 24.0):
        self.n_t = n_t
        self.t_max = t_max
        self.t = np.linspace(0.0, t_max, n_t)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 5)

        Returns
        -------
        C : (n, n_t) — concentration [mg/L]
        """
        D, k10, k12, k21, V1 = [X[:, i] for i in range(5)]
        t = self.t[None, :]  # (1, n_t)

        # Eigenvalues of the 2-compartment system
        s = k10 + k12 + k21
        disc = np.sqrt(np.maximum(s**2 - 4.0 * k10 * k21, 0.0))
        alpha = 0.5 * (s + disc)  # (n,) fast decay
        beta = 0.5 * (s - disc)  # (n,) slow decay
        beta = np.maximum(beta, 1e-10)

        DV = (D / V1)[:, None]  # (n, 1)
        ab_diff = (alpha - beta)[:, None]  # (n, 1)
        A = DV * (alpha[:, None] - k21[:, None]) / ab_diff
        B = DV * (k21[:, None] - beta[:, None]) / ab_diff

        C = A * np.exp(-alpha[:, None] * t) + B * np.exp(-beta[:, None] * t)
        return np.maximum(C, 0.0)
