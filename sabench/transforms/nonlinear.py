"""Nonlinear pointwise output transformations."""

from __future__ import annotations

import numpy as np


def t_softplus_pointwise(Y: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """Softplus pointwise transform: Z(z) = log(1 + exp(β · Y(z))) / β."""
    return np.log1p(np.exp(np.clip(beta * Y, -500, 500))) / beta


def t_cosh_pointwise(Y: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """Hyperbolic cosine: φ(y) = cosh(scale·y)."""
    return np.cosh(np.clip(scale * Y, -100, 100))


def t_cbrt_pointwise(Y: np.ndarray) -> np.ndarray:
    """Cube root: φ(y) = cbrt(y)."""
    return np.cbrt(Y)


def t_logistic_pointwise(Y: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Logistic: φ(y) = 1 / (1 + exp(-k·y))."""
    return 1.0 / (1.0 + np.exp(-np.clip(k * Y, -100, 100)))


def t_arctan_pointwise(Y: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Arctan: φ(y) = arctan(scale·y)."""
    return np.arctan(scale * Y)


def t_sinh_pointwise(Y: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """Sinh: φ(y) = sinh(scale·y)."""
    return np.sinh(np.clip(scale * Y, -100, 100))
