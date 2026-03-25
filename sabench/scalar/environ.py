"""
Environmental / ecological model — 4-input scalar benchmark.

Models the concentration S of a pollutant spill at a measuring point,
derived from the 1D convection-diffusion equation. Used in SA literature
to test sensitivity methods on environmental simulation models.

  S = M / sqrt(4π D t) * exp(-v^2 t / (4D)) * exp(-m / (4D/v)) [simplified]

More precisely, the Bliznyuk et al. (2008) formulation:

  S(x_obs, t_obs) = M / sqrt(4π D L) * exp(-(x_obs - v*L)^2 / (4*D*L))

  where L = t_obs is the observation time/length proxy.

References
----------
Bliznyuk, N., Ruppert, D., Shoemaker, C., Regis, R., Wild, S., & Mugunthan, P.
  (2008). Bayesian calibration and uncertainty analysis for computationally
  expensive models using optimization and radial basis function approximation.
  Journal of Computational and Graphical Statistics, 17(2), 270-294.
  https://doi.org/10.1198/106186008X320681

Surjanovic, S., & Bingham, D. (2013). Virtual Library of Simulation Experiments.
  https://www.sfu.ca/~ssurjano
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class EnvironModel(BenchmarkFunction):
    """
    Bliznyuk et al. (2008) 4-input environmental model.

    Models peak contaminant concentration at x=0, t=L.

    Inputs
    ------
    M  : mass of pollutant spilled [g]  U[7,   13]
    D  : diffusion coefficient [m²/s]  U[0.02, 0.12]
    L  : length of river reach [m]     U[0.01, 3]
    τ  : time of measurement [s]       U[30.01, 30.295]
    """

    name = "EnvironModel"
    d = 4
    output_type = "scalar"
    description = "Pollutant spill in 1D convection-diffusion model. No closed-form Sobol indices."
    reference = "Bliznyuk et al. (2008), JCGS 17(2). doi:10.1198/106186008X320681"

    bounds = [
        (7.0, 13.0),  # M
        (0.02, 0.12),  # D
        (0.01, 3.0),  # L
        (30.01, 30.295),  # tau
    ]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        M, D, L, tau = (X[:, i] for i in range(4))
        # Concentration at x=0, t=tau via 1D Gaussian plume
        c1 = M / np.sqrt(4.0 * np.pi * D * tau)
        c2 = np.exp(-(L**2) / (4.0 * D * tau))
        # Add image source at -2L for reflecting boundary at x=0
        c3 = np.exp(-((2 * L) ** 2) / (4.0 * D * tau))
        C = c1 * (c2 + c3)
        return np.sqrt(4.0 * np.pi) * C
