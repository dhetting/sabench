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


def t_cube_pointwise(Y: np.ndarray) -> np.ndarray:
    """Cube: φ(y) = y³."""
    return Y**3


def t_erf_pointwise(Y: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Error function: φ(y) = erf(scale·y)."""
    from math import erf as _erf

    return np.vectorize(lambda y: _erf(scale * y))(Y).astype(float)


def t_sin_pointwise(Y: np.ndarray, freq: float = 0.5) -> np.ndarray:
    """Sine: φ(y) = sin(freq·y)."""
    return np.sin(freq * Y)


def t_cos_pointwise(Y: np.ndarray, freq: float = 0.5) -> np.ndarray:
    """Cosine: φ(y) = cos(freq·y)."""
    return np.cos(freq * Y)


def t_step_pointwise(Y: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Heaviside step: φ(y) = 1[y > threshold]."""
    return (Y > threshold).astype(float)


def t_log_abs(Y: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Log of shifted absolute value: φ(y) = log(|y| + eps)."""
    return np.log(np.abs(Y) + eps)


def t_sinc(Y: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Normalised sinc: φ(y) = sin(π·scale·y) / (π·scale·y)."""
    return np.sinc(scale * Y)


def t_sin_squared(Y: np.ndarray, freq: float = 0.5) -> np.ndarray:
    """Squared sine: φ(y) = sin²(freq·y)."""
    return np.sin(freq * Y) ** 2


def t_cos_squared(Y: np.ndarray, freq: float = 0.5) -> np.ndarray:
    """Squared cosine: φ(y) = cos²(freq·y)."""
    return np.cos(freq * Y) ** 2


def t_damped_sin(Y: np.ndarray, freq: float = 0.5, decay: float = 0.1) -> np.ndarray:
    """Damped sine: φ(y) = exp(-decay·|y|) · sin(freq·y)."""
    return np.exp(-decay * np.abs(Y)) * np.sin(freq * Y)


def t_sawtooth(Y: np.ndarray, period: float = 4.0) -> np.ndarray:
    """Sawtooth wave: φ(y) = 2·(y/period - floor(y/period + 0.5))."""
    return 2.0 * (Y / period - np.floor(Y / period + 0.5))


def t_square_wave(Y: np.ndarray, period: float = 4.0) -> np.ndarray:
    """Square wave: φ(y) = sign(sin(2π·y/period))."""
    return np.sign(np.sin(2.0 * np.pi * Y / period))


def t_double_sin(Y: np.ndarray, freq1: float = 0.3, freq2: float = 0.7) -> np.ndarray:
    """Double sine: φ(y) = sin(freq1·y) + sin(freq2·y)."""
    return np.sin(freq1 * Y) + np.sin(freq2 * Y)


def t_sin_cos_product(Y: np.ndarray, freq: float = 0.5) -> np.ndarray:
    """Harmonic product: φ(y) = sin(freq·y) · cos(freq·y)."""
    return np.sin(freq * Y) * np.cos(freq * Y)
