"""Environmental and hydrological transform implementations."""

from __future__ import annotations

import numpy as np

from sabench.transforms.utilities import _bc, _ymin


def t_growing_degree_days(Y: np.ndarray, base: float = 10.0) -> np.ndarray:
    """Growing degree days: phi = max(Y - base, 0) -- threshold accumulation."""
    return np.maximum(Y - base, 0.0)


def t_standardised_precip_idx(Y: np.ndarray) -> np.ndarray:
    """Standardised precip index proxy: phi = (Y - mean)/std -- climate anomaly."""
    flat = Y.reshape(len(Y), -1)
    mu = _bc(flat.mean(axis=1), Y)
    sg = _bc(flat.std(axis=1) + 1e-10, Y)
    return (Y - mu) / sg


def t_nash_sutcliffe(Y: np.ndarray) -> np.ndarray:
    """Nash-Sutcliffe efficiency proxy: phi = 1 - MSE/Var(Y_obs) -- goodness-of-fit."""
    flat = Y.reshape(len(Y), -1)
    mu = _bc(flat.mean(axis=1), Y)
    var_obs = _bc(flat.var(axis=1) + 1e-12, Y)
    mse = (Y - mu) ** 2
    return 1.0 - mse / var_obs


def t_pot_log(Y: np.ndarray, q: float = 0.90, eps: float = 1.0) -> np.ndarray:
    """Log-transformed peaks-over-threshold: phi = log(max(Y-threshold, 0)+eps)."""
    threshold = _bc(np.quantile(Y.reshape(len(Y), -1), q, axis=1), Y)
    return np.log(np.maximum(Y - threshold, 0.0) + eps)


def t_log_flow(Y: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """Log streamflow: phi = log(Y - ymin + eps) -- standard hydrological transform."""
    s = _bc(_ymin(Y), Y)
    return np.log(Y - s + eps)
