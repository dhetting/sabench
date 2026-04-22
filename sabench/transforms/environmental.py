"""Environmental and hydrological transform implementations."""

from __future__ import annotations

import numpy as np

from sabench.transforms.utilities import _bc, _safe_range, _ymin


def t_log_shift(Y: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Shifted log transform for nonnegative environmental responses."""
    return np.log(Y - _bc(_ymin(Y), Y) + eps)


def t_power_law(Y: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Normalised power-law response for environmental magnitude scaling."""
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    return ((Y - s) / r) ** beta


def t_box_cox(Y: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """Box-Cox transform on a shifted positive support."""
    s = _bc(_ymin(Y), Y)
    y_pos = Y - s + 1.0
    return np.log(y_pos) if abs(lam) < 1e-8 else (y_pos**lam - 1.0) / lam


def t_clipped_excess(Y: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    """Clipped excess above a sample quantile threshold."""
    threshold = _bc(np.quantile(Y.reshape(len(Y), -1), quantile, axis=1), Y)
    return np.maximum(Y - threshold, 0.0)


def _exceed(Y: np.ndarray, q: float) -> np.ndarray:
    """Indicator exceedance above a sample quantile threshold."""
    threshold = _bc(np.quantile(Y.reshape(len(Y), -1), q, axis=1), Y)
    return (Y > threshold).astype(float)


def t_exceed_q75(Y: np.ndarray) -> np.ndarray:
    """Indicator for exceedance above the 75th percentile."""
    return _exceed(Y, 0.75)


def t_exceed_q90(Y: np.ndarray) -> np.ndarray:
    """Indicator for exceedance above the 90th percentile."""
    return _exceed(Y, 0.90)


def t_exceed_q95(Y: np.ndarray) -> np.ndarray:
    """Indicator for exceedance above the 95th percentile."""
    return _exceed(Y, 0.95)


def t_exceed_q99(Y: np.ndarray) -> np.ndarray:
    """Indicator for exceedance above the 99th percentile."""
    return _exceed(Y, 0.99)


def t_log2_shift(Y: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Base-2 shifted log transform for scale compression."""
    return np.log2(Y - _bc(_ymin(Y), Y) + eps)


def t_log10_shift(Y: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Base-10 shifted log transform for scale compression."""
    return np.log10(Y - _bc(_ymin(Y), Y) + eps)


def t_log_log(Y: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Nested shifted log transform for heavy-tailed compression."""
    shifted = Y - _bc(_ymin(Y), Y) + eps
    return np.log(np.log(shifted + 1.0) + 1.0)


def t_anomaly_pct(Y: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Anomaly percent: phi = (Y - mean)/|mean| -- percentage departure from mean."""
    flat = Y.reshape(len(Y), -1)
    mu = _bc(flat.mean(axis=1), Y)
    denom = np.abs(mu) + eps
    return (Y - mu) / denom


def t_bias_correction(Y: np.ndarray) -> np.ndarray:
    """Bias correction (linear scaling): phi = Y * (target_mean/sample_mean)."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1) + 1e-12
    return Y * _bc(1.0 / mu, Y)


def t_quantile_delta(Y: np.ndarray, q: float = 0.90) -> np.ndarray:
    """Quantile delta: phi = Y * (q_target/q_sample) -- quantile scaling correction."""
    flat = Y.reshape(len(Y), -1)
    qvals = np.quantile(flat, q, axis=1)
    scale = 1.0 / (qvals + 1e-10)
    return Y * _bc(scale, Y)


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
