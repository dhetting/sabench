"""
Borehole function — 8-input scalar benchmark for groundwater flow.

Models the flow rate (m³/yr) through a borehole in the ground.
Widely used in SA literature because of its extreme sensitivity concentration
(~4/8 inputs dominate almost all variance).

  w = 2π·Tm·(Hu - Hl) / (ln(r/rw) · (1 + 2L·Tm/(ln(r/rw)·rw²·Kw) + Tm/Tl))

References
----------
Harper, W. V., & Gupta, S. K. (1983). Sensitivity/uncertainty analysis of a
  borehole scenario comparing Latin hypercube sampling and deterministic
  sensitivity analysis. Technical Report BMI/ONWI-516. Battelle Memorial Institute.

Morris, M. D., Mitchell, T. J., & Ylvisaker, D. (1993). Bayesian design and
  analysis of computer experiments: Use of derivatives in surface prediction.
  Technometrics, 35(3), 243-255. https://doi.org/10.2307/1269888

Surjanovic, S., & Bingham, D. (2013). Virtual Library of Simulation Experiments.
  Simon Fraser University. https://www.sfu.ca/~ssurjano
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class Borehole(BenchmarkFunction):
    """
    Borehole groundwater flow rate (d=8).

    Inputs
    ------
    rw  : borehole radius [m]          N(0.10, 0.0161812)  clipped to [0.05, 0.15]
    r   : radius of influence [m]      LogN(7.71, 1.0056)  clipped to [100, 50000]
    Tu  : transmissivity upper [m²/yr] U[63070, 115600]
    Hu  : potentiometric head upper [m] U[990, 1110]
    Tl  : transmissivity lower [m²/yr] U[63.1, 116]
    Hl  : potentiometric head lower [m] U[700, 820]
    L   : length of borehole [m]       U[1120, 1680]
    Kw  : hydraulic conductivity [m/yr] U[9855, 12045]

    Note: bounds given as the bracketing U distributions for reproducibility
    (using uniform approximations to the original normal/lognormal inputs
    as standard in the SA literature, following Surjanovic & Bingham 2013).
    """

    name = "Borehole"
    d = 8
    output_type = "scalar"
    description = "Groundwater flow; 2 inputs dominant. No simple closed-form Sobol indices."
    reference = "Harper & Gupta (1983), BMI/ONWI-516; Morris et al. (1993), Technometrics 35(3)."

    bounds = [
        (0.05, 0.15),  # rw
        (100.0, 50000.0),  # r
        (63070.0, 115600.0),  # Tu
        (990.0, 1110.0),  # Hu
        (63.1, 116.0),  # Tl
        (700.0, 820.0),  # Hl
        (1120.0, 1680.0),  # L
        (9855.0, 12045.0),  # Kw
    ]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 8)

        Returns
        -------
        w : (n,) — flow rate [m³/yr]
        """
        rw, r, Tu, Hu, Tl, Hl, L, Kw = [X[:, i] for i in range(8)]
        log_r_rw = np.log(r / rw)
        numer = 2.0 * np.pi * Tu * (Hu - Hl)
        denom = log_r_rw * (1.0 + 2.0 * L * Tu / (log_r_rw * rw**2 * Kw) + Tu / Tl)
        return numer / denom
