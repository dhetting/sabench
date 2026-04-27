"""Engineering reliability and failure transforms extracted from the legacy monolith."""

from __future__ import annotations

import numpy as np

from sabench.transforms.utilities import _bc, _safe_range, _ymin


def t_carnot_quadratic(Y: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Quadratic transform after per-sample Carnot-style normalization."""
    shift = _bc(_ymin(Y), Y)
    value_range = _bc(_safe_range(Y), Y)
    return ((Y - shift + delta) / (value_range + delta)) ** 2


def t_arrhenius(Y: np.ndarray, Ea_over_R: float = 2.0) -> np.ndarray:
    """Arrhenius-type exponential transform on positive shifted support."""
    shift = _bc(_ymin(Y), Y)
    y_pos = Y - shift + 1.0
    return np.exp(-Ea_over_R / y_pos)


def t_normalised_stress(Y: np.ndarray, yield_q: float = 0.80) -> np.ndarray:
    """Goodman-style normalized stress clipped to the unit interval."""
    shift = _bc(_ymin(Y), Y)
    yield_level = _bc(np.quantile(Y.reshape(len(Y), -1), yield_q, axis=1), Y)
    return np.clip((Y - shift) / (yield_level - shift + 1e-12), 0.0, 1.0)


def t_weibull_reliability(Y: np.ndarray, shape: float = 2.0) -> np.ndarray:
    """Weibull-style failure probability on per-sample normalized support."""
    shift = _bc(_ymin(Y), Y)
    value_range = _bc(_safe_range(Y), Y)
    return 1.0 - np.exp(-(((Y - shift) / value_range) ** shape))


def t_fatigue_miner(Y: np.ndarray, m: float = 3.0) -> np.ndarray:
    """Palmgren-Miner damage accumulation on normalized amplitude."""
    shift = _bc(_ymin(Y), Y)
    value_range = _bc(_safe_range(Y), Y)
    y_norm = (Y - shift) / value_range + 1e-6
    return y_norm**m


def t_rankine_failure(Y: np.ndarray) -> np.ndarray:
    """Rankine failure criterion using the positive principal-stress proxy."""
    flat = Y.reshape(len(Y), -1)
    return np.maximum(flat, 0.0).reshape(Y.shape)


def t_von_mises(Y: np.ndarray) -> np.ndarray:
    """Von Mises stress proxy as a scaled absolute-value response."""
    return np.abs(Y) * np.sqrt(1.25)


def t_safety_factor(Y: np.ndarray, capacity: float = 1.0) -> np.ndarray:
    """Safety factor as inverse stress magnitude relative to a capacity."""
    return capacity / (np.abs(Y) + 1e-6)


def t_cumulative_damage(Y: np.ndarray, m: float = 3.0) -> np.ndarray:
    """Palmgren-Miner cumulative damage under a power-law stress response."""
    return np.abs(Y) ** m


def t_stress_life(Y: np.ndarray, C: float = 1e6, m: float = 3.0) -> np.ndarray:
    """Basquin S-N stress-life relation."""
    return C / (np.abs(Y) + 1e-3) ** m
