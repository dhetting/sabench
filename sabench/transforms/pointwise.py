"""Pointwise output transformations."""

from __future__ import annotations

import numpy as np


def t_tanh_pointwise(Y: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Hyperbolic tangent saturation: Z(z) = tanh(α · Y(z))."""
    return np.tanh(alpha * Y)
