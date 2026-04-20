"""
SIR epidemic model — 3-input functional benchmark.

  dS/dt = -β S I / N
  dI/dt =  β S I / N - γ I
  dR/dt =  γ I

Models the spread of an infectious disease in a closed population.
Used in SA literature to test sensitivity methods on compartmental epidemiological models.

References
----------
Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical
  theory of epidemics. Proceedings of the Royal Society of London A, 115(772),
  700-721. https://doi.org/10.1098/rspa.1927.0118

Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S.
  (2010). Variance based sensitivity analysis of model output. Design and estimator
  for the total sensitivity index. Computer Physics Communications, 181(2), 259-270.
  https://doi.org/10.1016/j.cpc.2009.09.018
"""

from __future__ import annotations

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class EpidemicSIR(BenchmarkFunction):
    """
    SIR epidemic model (d=3, functional output).

    Inputs
    ------
    beta  : transmission rate          U[0.1, 0.5]
    gamma : recovery rate              U[0.05, 0.2]
    I0    : initial infected fraction  U[0.001, 0.01]

    Output: Infected fraction I(t)/N at n_t time steps.
    """

    name = "EpidemicSIR"
    d = 3
    output_type = "functional"
    description = "SIR compartmental model; R0=β/γ governs outbreak. No analytical Sobol indices."
    reference = "Kermack & McKendrick (1927), Proc. R. Soc. A 115(772). doi:10.1098/rspa.1927.0118"

    bounds = [
        (0.1, 0.5),  # beta
        (0.05, 0.2),  # gamma
        (0.001, 0.01),  # I0 (fraction)
    ]

    def __init__(self, n_t: int = 100, t_max: float = 100.0, N: float = 1.0):
        self.n_t = n_t
        self.t_max = t_max
        self.N = N  # total population (normalised to 1)
        self.t = np.linspace(0.0, t_max, n_t)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 3)

        Returns
        -------
        infected : (n, n_t) — infected fraction I(t)/N
        """
        n = len(X)
        beta, gamma, I0 = X[:, 0], X[:, 1], X[:, 2]
        dt = self.t_max / (self.n_t - 1)

        susceptible = 1.0 - I0
        infected = I0.copy()
        recovered = np.zeros(n)

        out = np.empty((n, self.n_t))
        out[:, 0] = infected

        for s in range(1, self.n_t):
            d_susceptible = -beta * susceptible * infected
            d_infected = beta * susceptible * infected - gamma * infected
            d_recovered = gamma * infected
            susceptible = np.maximum(susceptible + dt * d_susceptible, 0.0)
            infected = np.maximum(infected + dt * d_infected, 0.0)
            recovered = recovered + dt * d_recovered
            out[:, s] = infected

        return out
