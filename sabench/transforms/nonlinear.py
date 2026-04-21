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


def t_gompertz(Y: np.ndarray, b: float = 1.0, c: float = 0.5) -> np.ndarray:
    """Gompertz CDF: φ(y) = exp(-exp(-b·(u-c)))."""
    shift = float(np.nanmin(Y))
    value_range = float(np.nanmax(Y) - shift)
    if not np.isfinite(value_range) or value_range <= 0.0:
        value_range = 1.0
    u = (Y - shift) / value_range - c
    return np.exp(-np.exp(-b * u))


def t_algebraic_sigmoid(Y: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Algebraic sigmoid: φ(y) = u / sqrt(1 + u²)."""
    u = scale * Y
    return u / np.sqrt(1.0 + u**2)


def t_swish(Y: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Swish activation: φ(y) = y · sigmoid(beta·y)."""
    return Y * (1.0 / (1.0 + np.exp(-np.clip(beta * Y, -100, 100))))


def t_mish(Y: np.ndarray) -> np.ndarray:
    """Mish activation: φ(y) = y · tanh(softplus(y))."""
    softplus = np.log1p(np.exp(np.clip(Y, -500, 500)))
    return Y * np.tanh(softplus)


def t_selu(Y: np.ndarray, alpha: float = 1.6733, lam: float = 1.0507) -> np.ndarray:
    """SELU activation: scaled ELU."""
    return lam * np.where(Y >= 0.0, Y, alpha * (np.exp(np.clip(Y, -100, 100)) - 1.0))


def t_softsign(Y: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Softsign: φ(y) = scale·y / (1 + |y|)."""
    return scale * Y / (1.0 + np.abs(Y))


def t_bent_identity(Y: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Bent identity: φ(y) = (sqrt(u²+1)-1)/2 + u."""
    u = scale * Y
    return (np.sqrt(u**2 + 1.0) - 1.0) / 2.0 + u


def t_hard_sigmoid(Y: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Hard sigmoid: φ(y) = clip(0.2·scale·y + 0.5, 0, 1)."""
    return np.clip(0.2 * scale * Y + 0.5, 0.0, 1.0)


def t_hard_tanh(Y: np.ndarray, scale: float = 0.3) -> np.ndarray:
    """Hard tanh: φ(y) = clip(scale·y, -1, 1)."""
    return np.clip(scale * Y, -1.0, 1.0)


def t_soft_threshold(Y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Soft threshold: φ(y) = sign(y) · max(|y| - λ, 0)."""
    return np.sign(Y) * np.maximum(np.abs(Y) - lam, 0.0)


def t_hard_threshold(Y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Hard threshold: φ(y) = y · 1[|y| ≥ λ]."""
    return Y * (np.abs(Y) >= lam).astype(float)


def t_ramp(Y: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Ramp / clipped linear map: φ(y) = clip(y, lo, hi)."""
    return np.clip(Y, lo, hi)


def t_spike(Y: np.ndarray, center: float = 0.0, width: float = 1.0) -> np.ndarray:
    """Gaussian spike: φ(y) = exp(-0.5 · ((y - c) / w)^2)."""
    return np.exp(-0.5 * ((Y - center) / width) ** 2)


def t_breakpoint(
    Y: np.ndarray,
    bp: float = 0.0,
    slope_lo: float = 0.5,
    slope_hi: float = 2.0,
) -> np.ndarray:
    """Piecewise linear breakpoint with a C0 kink at ``bp``."""
    return np.where(Y < bp, slope_lo * (Y - bp), slope_hi * (Y - bp))


def t_hockey_stick(Y: np.ndarray, bp: float = 0.0) -> np.ndarray:
    """Hockey stick: φ(y) = max(y - bp, 0)."""
    return np.maximum(Y - bp, 0.0)


def t_deadzone(Y: np.ndarray, half_width: float = 1.0) -> np.ndarray:
    """Deadzone: φ(y) = 0 for |y| < h, otherwise sign(y) · (|y| - h)."""
    return np.sign(Y) * np.maximum(np.abs(Y) - half_width, 0.0)
