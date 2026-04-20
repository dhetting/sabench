"""Pointwise output transformations."""

from __future__ import annotations

import numpy as np


def t_affine(Y: np.ndarray, a: float = 2.0, b: float = 1.0) -> np.ndarray:
    """Affine (linear) pointwise transform: Z(z) = a · Y(z) + b.

    This is the canonical commutative case for Sobol indices.
    """
    return a * Y + b


def t_tanh_pointwise(Y: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Hyperbolic tangent saturation: Z(z) = tanh(α · Y(z))."""
    return np.tanh(alpha * Y)
