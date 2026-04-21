"""
Saltelli (2002) quasi-random sample design.

Produces the row-block matrix [A | B | AB_1 | ... | AB_d]
needed by the Jansen (1999) first-order and total-effect estimators.
"""

from __future__ import annotations

import numpy as np


def saltelli_sample(
    d: int,
    bounds: list,
    N: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate Saltelli sample design of size N*(d+2).

    Parameters
    ----------
    d      : number of model inputs
    bounds : list of (lo, hi) pairs, length d
    N      : base sample size (total evaluations = N*(d+2))
    seed   : random seed for reproducibility

    Returns
    -------
    X : ndarray (N*(d+2), d)
        Stacked [A; B; AB_1; ...; AB_d]
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    A = rng.random((N, d)) * (hi - lo) + lo
    B = rng.random((N, d)) * (hi - lo) + lo
    blocks = [A, B]
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        blocks.append(AB)
    return np.vstack(blocks)
