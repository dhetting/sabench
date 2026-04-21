"""Pointwise output transformations."""

from __future__ import annotations

import numpy as np


def t_tanh_pointwise(Y: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Hyperbolic tangent saturation: Z(z) = tanh(α · Y(z))."""
    return np.tanh(alpha * Y)


def t_square_pointwise(Y: np.ndarray) -> np.ndarray:
    """Square: φ(y) = y²."""
    return Y**2


def t_exp_pointwise(Y: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """Exponential: φ(y) = exp(scale·y)."""
    return np.exp(np.clip(scale * Y, -100, 100))


def t_relu_pointwise(Y: np.ndarray) -> np.ndarray:
    """ReLU: φ(y) = max(0, y)."""
    return np.maximum(Y, 0.0)


def t_log1p_abs(Y: np.ndarray) -> np.ndarray:
    """Signed log(1 + |y|) transform."""
    return np.log1p(np.abs(Y)) * np.sign(Y)


def t_sqrt_abs(Y: np.ndarray) -> np.ndarray:
    """Signed square-root transform."""
    return np.sqrt(np.abs(Y)) * np.sign(Y)


def t_abs_pointwise(Y: np.ndarray) -> np.ndarray:
    """Absolute value transform."""
    return np.abs(Y)
