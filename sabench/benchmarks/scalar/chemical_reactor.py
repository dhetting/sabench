"""
CSTR (Continuous Stirred Tank Reactor) — 5-input scalar benchmark.

Models steady-state outlet concentration from a first-order exothermic
reaction in a CSTR. The Damköhler number and heat generation term create
strong nonlinearity and potential multiplicity.

  X_out = X_in / (1 + Da * exp(-E_a / (R * T)))

where T satisfies the energy balance:

  T = T_feed + Delta_T_ad * X_in * Da * exp(-E_a/(R*T)) / (1 + Da * exp(-E_a/(R*T)))

(solved iteratively)

References
----------
Saltelli, A., & Tarantola, S. (2002). On the relative importance of input
  factors in mathematical models: Safety assessment for nuclear waste disposal.
  Journal of the American Statistical Association, 97(459), 702-709.
  https://doi.org/10.1198/016214502388618447

Blatman, G., & Sudret, B. (2010). An adaptive algorithm to build up sparse
  polynomial chaos expansions for stochastic finite element analysis.
  Probabilistic Engineering Mechanics, 25(2), 183-197.
"""

from __future__ import annotations

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class CSTRReactor(BenchmarkFunction):
    """
    CSTR steady-state conversion (d=5).

    Inputs
    ------
    X_in    : inlet mole fraction       U[0.5, 0.8]
    T_feed  : feed temperature [K]      U[300, 400]
    Da      : Damköhler number          U[0.5, 5.0]
    Ea_over_R : E_a/R [K]              U[1000, 6000]
    dT_ad   : adiabatic temperature rise U[20, 120]
    """

    name = "CSTRReactor"
    d = 5
    output_type = "scalar"
    description = (
        "CSTR steady-state conversion; Arrhenius nonlinearity. No analytical Sobol indices."
    )
    reference = "Saltelli & Tarantola (2002), JASA 97(459). doi:10.1198/016214502388618447"

    bounds = [
        (0.5, 0.8),  # X_in
        (300.0, 400.0),  # T_feed
        (0.5, 5.0),  # Da
        (1000.0, 6000.0),  # Ea/R
        (20.0, 120.0),  # dT_ad
    ]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X_in, T_feed, Da, EaR, dT_ad = (X[:, i] for i in range(5))

        # Solve energy balance T = T_feed + dT_ad * conversion iteratively
        T = T_feed.copy()
        for _ in range(20):
            k = np.exp(-EaR / T)
            conv = Da * k / (1.0 + Da * k)
            T = T_feed + dT_ad * conv
        k = np.exp(-EaR / T)
        conv = Da * k / (1.0 + Da * k)
        return X_in * conv
