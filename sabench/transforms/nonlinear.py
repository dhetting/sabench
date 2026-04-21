"""Nonlinear pointwise output transformations."""

from __future__ import annotations

import numpy as np


def t_softplus_pointwise(Y: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """Softplus pointwise transform: Z(z) = log(1 + exp(β · Y(z))) / β."""
    return np.log1p(np.exp(np.clip(beta * Y, -500, 500))) / beta
