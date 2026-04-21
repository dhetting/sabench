"""
Variance-based sensitivity estimators.

Jansen (1999) first-order and total-effect indices, fully vectorised
over any number of model outputs (scalar, spatial pixels, time steps, etc.).

Reference
---------
Jansen, M. J. W. (1999). Analysis of variance designs for model output.
  Computer Physics Communications, 117(1-2), 35-43.

Saltelli, A. (2002). Making best use of model evaluations to compute
  sensitivity indices. Computer Physics Communications, 145(2), 280-297.
"""

from __future__ import annotations

import numpy as np


def jansen_s1_st(
    Y: np.ndarray,
    N: int,
    d: int,
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jansen (1999) first-order and total-effect Sobol indices.

    Parameters
    ----------
    Y    : ndarray (N*(d+2),) or (N*(d+2), n_outputs)
           Model evaluations from a Saltelli sample design.
    N    : base sample size
    d    : number of inputs
    clip : if True, clip indices to [0, 1]

    Returns
    -------
    S1 : ndarray (d,) or (d, n_outputs) — first-order indices
    ST : ndarray (d,) or (d, n_outputs) — total-effect indices
    """
    scalar_out = Y.ndim == 1
    if scalar_out:
        Y = Y[:, None]

    n_out = Y.shape[1]
    YA = Y[:N]
    YB = Y[N : 2 * N]

    f0 = 0.5 * (YA.mean(0) + YB.mean(0))
    Var = 0.5 * (((YA - f0) ** 2).mean(0) + ((YB - f0) ** 2).mean(0))
    Var = np.where(Var > 0, Var, 1.0)  # guard zero-variance outputs

    S1 = np.empty((d, n_out))
    ST = np.empty((d, n_out))

    for i in range(d):
        YABi = Y[(2 + i) * N : (3 + i) * N]
        # Jansen (1999) estimators
        S1[i] = 1.0 - ((YB - YABi) ** 2).mean(0) / (2.0 * Var)
        ST[i] = ((YA - YABi) ** 2).mean(0) / (2.0 * Var)

    if clip:
        S1 = np.clip(S1, 0.0, 1.0)
        ST = np.clip(ST, 0.0, 1.0)

    if scalar_out:
        return S1[:, 0], ST[:, 0]
    return S1, ST


# Convenience aliases
def first_order(Y: np.ndarray, N: int, d: int) -> np.ndarray:
    """Return only first-order indices."""
    return jansen_s1_st(Y, N, d)[0]


def total_effect(Y: np.ndarray, N: int, d: int) -> np.ndarray:
    """Return only total-effect indices."""
    return jansen_s1_st(Y, N, d)[1]
