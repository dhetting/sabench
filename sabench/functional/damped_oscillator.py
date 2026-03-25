"""
Damped oscillator — 6-input functional benchmark.

Second-order linear ODE:
  m * u'' + c * u' + k * u = F0 * cos(ω_0 * t),   u(0) = 0, u'(0) = 0

Analytical solution available; closed-form Sobol indices can be derived
for the peak response. Widely used to benchmark functional sensitivity analysis.

References
----------
Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions.
  Reliability Engineering & System Safety, 93(7), 964-979.
  https://doi.org/10.1016/j.ress.2007.04.002

Marelli, S., & Sudret, B. (2018). UQLab: A framework for uncertainty quantification
  in Matlab. In Proceedings of the 2nd International Conference on Vulnerability
  and Risk Analysis and Management. https://doi.org/10.1061/9780784413609.257
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction


class DampedOscillator(BenchmarkFunction):
    """
    Damped harmonic oscillator (d=6, functional output over time).

    Inputs
    ------
    m   : mass [kg]                U[1,  1.5]
    c   : damping coefficient      U[0.1, 0.5]
    k   : stiffness [N/m]          U[0.5, 1.5]
    F0  : forcing amplitude [N]    U[0.5, 1.5]
    omega_0 : forcing frequency    U[0.5, 1.5]
    t0  : observation start [s]    U[0,  0.5]   (unused, for d=6 compat)

    Output: displacement u(t) at n_t time steps.
    """

    name = "DampedOscillator"
    d = 6
    output_type = "functional"
    description = (
        "2nd-order ODE with harmonic forcing; analytical solution. "
        "Temporal output with peak-response analytical Sobol indices."
    )
    reference = "Sudret (2008), RESS 93(7). doi:10.1016/j.ress.2007.04.002"

    bounds = [
        (1.0, 1.5),  # m
        (0.1, 0.5),  # c
        (0.5, 1.5),  # k
        (0.5, 1.5),  # F0
        (0.5, 1.5),  # omega_0
        (0.0, 0.5),  # t0 (dummy 6th input for d=6 compat)
    ]

    def __init__(self, n_t: int = 100, t_max: float = 20.0):
        self.n_t = n_t
        self.t_max = t_max
        self.t = np.linspace(0.0, t_max, n_t)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n, 6)

        Returns
        -------
        Y : (n, n_t) — displacement time series
        """
        m, c, k, F0, omega0, _ = (X[:, i] for i in range(6))
        t = self.t[None, :]  # (1, n_t)

        omega_n = np.sqrt(k / m)[:, None]  # natural frequency (n,1)
        zeta = (c / (2.0 * np.sqrt(k * m)))[:, None]  # damping ratio
        omega_d = omega_n * np.sqrt(np.maximum(1.0 - zeta**2, 1e-8))  # damped freq

        F0 = F0[:, None]
        om0 = omega0[:, None]
        m_bc = m[:, None]

        # Particular solution: u_p = X_p cos(omega0 t) + Y_p sin(omega0 t)
        denom = (omega_n**2 - om0**2) ** 2 + (2.0 * zeta * omega_n * om0) ** 2
        denom = np.maximum(denom, 1e-12)
        X_p = F0 / m_bc * (omega_n**2 - om0**2) / denom
        Y_p = F0 / m_bc * (2.0 * zeta * omega_n * om0) / denom

        # Homogeneous solution: underdamped case
        sigma = zeta * omega_n
        u_hom_coeff_cos = -X_p  # from u(0) = 0
        u_hom_coeff_sin = (sigma * X_p - om0 * Y_p) / omega_d  # from u'(0) = 0

        u = (
            np.exp(-sigma * t)
            * (u_hom_coeff_cos * np.cos(omega_d * t) + u_hom_coeff_sin * np.sin(omega_d * t))
            + X_p * np.cos(om0 * t)
            + Y_p * np.sin(om0 * t)
        )
        return u
