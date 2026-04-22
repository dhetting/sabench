"""Aggregation transforms extracted from the legacy transform monolith."""

from __future__ import annotations

import numpy as np

from sabench.transforms.utilities import _bc


def t_temporal_peak(Y: np.ndarray) -> np.ndarray:
    """Scalar peak value max_t Y(t).

    Reduces a time series to its peak response, which is the most common
    engineering summary statistic (peak demand, maximum flood stage, peak
    ground acceleration). Compressing n_t dimensions to 1 is a severe
    reduction that destroys temporal structure and hence strongly alters
    which inputs appear dominant.
    """
    flat = Y.reshape(len(Y), -1)
    peak = flat.max(axis=1)
    return (peak[:, None] * np.ones_like(flat)).reshape(Y.shape)



def t_temporal_rms(Y: np.ndarray) -> np.ndarray:
    """Root-mean-square temporal summary rebroadcast to the original shape."""
    flat = Y.reshape(len(Y), -1)
    rms = np.sqrt((flat**2).mean(axis=1))
    return _bc(rms, Y) * np.ones_like(Y)


def t_temporal_range(Y: np.ndarray) -> np.ndarray:
    """Temporal range summary rebroadcast to the original shape."""
    flat = Y.reshape(len(Y), -1)
    spread = flat.max(axis=1) - flat.min(axis=1)
    return _bc(spread, Y) * np.ones_like(Y)


def t_temporal_autocorr(Y: np.ndarray) -> np.ndarray:
    """Lag-1 temporal autocorrelation rebroadcast to the original shape."""
    flat = Y.reshape(len(Y), -1)
    n_steps = flat.shape[1]
    if n_steps < 3:
        return np.zeros_like(Y)
    mean = flat.mean(axis=1, keepdims=True)
    variance = flat.var(axis=1).clip(min=1e-12)
    autocorr = ((flat[:, :-1] - mean) * (flat[:, 1:] - mean)).mean(axis=1) / variance
    return _bc(autocorr, Y) * np.ones_like(Y)


def t_temporal_quantile(Y: np.ndarray, q: float = 0.50) -> np.ndarray:
    """Temporal quantile summary rebroadcast to the original shape."""
    flat = Y.reshape(len(Y), -1)
    quantile = np.quantile(flat, q, axis=1)
    return _bc(quantile, Y) * np.ones_like(Y)

def t_sample_variance(Y: np.ndarray) -> np.ndarray:
    """Sample variance of outputs across pixels or time."""
    flat = Y.reshape(len(Y), -1)
    vv = flat.var(axis=1)
    return _bc(vv, Y) * np.ones_like(Y)


def t_sample_skewness(Y: np.ndarray) -> np.ndarray:
    """Sample skewness as the third standardized central moment."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sg = flat.std(axis=1, keepdims=True) + 1e-10
    skew = ((flat - mu) ** 3).mean(axis=1) / sg.squeeze() ** 3
    return _bc(skew, Y) * np.ones_like(Y)


def t_sample_kurtosis(Y: np.ndarray) -> np.ndarray:
    """Excess kurtosis as the fourth standardized central moment minus three."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sg = flat.std(axis=1, keepdims=True) + 1e-10
    kurt = ((flat - mu) ** 4).mean(axis=1) / sg.squeeze() ** 4 - 3.0
    return _bc(kurt, Y) * np.ones_like(Y)


def t_percentile_q10(Y: np.ndarray) -> np.ndarray:
    """10th percentile summary rebroadcast to the original output shape."""
    flat = Y.reshape(len(Y), -1)
    q = np.percentile(flat, 10, axis=1)
    return _bc(q, Y) * np.ones_like(Y)


def t_percentile_q90(Y: np.ndarray) -> np.ndarray:
    """90th percentile summary rebroadcast to the original output shape."""
    flat = Y.reshape(len(Y), -1)
    q = np.percentile(flat, 90, axis=1)
    return _bc(q, Y) * np.ones_like(Y)


def t_interquartile_range(Y: np.ndarray) -> np.ndarray:
    """Interquartile range summary rebroadcast to the original output shape."""
    flat = Y.reshape(len(Y), -1)
    iqr = np.percentile(flat, 75, axis=1) - np.percentile(flat, 25, axis=1)
    return _bc(iqr, Y) * np.ones_like(Y)


def t_negentropy_proxy(Y: np.ndarray) -> np.ndarray:
    """Negentropy proxy rebroadcast to the original output shape."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sg = flat.std(axis=1, keepdims=True) + 1e-10
    u = (flat - mu) / sg
    kurt = (u**4).mean(axis=1) - 3.0
    mu_u = u.mean(axis=1)
    neg = (mu_u**2 + 0.25 * kurt**2) / 16.0
    return _bc(neg, Y) * np.ones_like(Y)


def t_wasserstein_proxy(Y: np.ndarray) -> np.ndarray:
    """Wasserstein-1 proxy rebroadcast to the original output shape."""
    flat = Y.reshape(len(Y), -1)
    med = np.median(flat, axis=1, keepdims=True)
    mad = np.abs(flat - med).mean(axis=1)
    return _bc(mad, Y) * np.ones_like(Y)


def t_energy_distance_proxy(Y: np.ndarray) -> np.ndarray:
    """Energy-distance proxy rebroadcast to the original output shape."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sigma = flat.std(axis=1, keepdims=True) + 1e-10
    u = (flat - mu) / sigma
    ed = np.abs(u).mean(axis=1)
    return _bc(ed, Y) * np.ones_like(Y)


def t_entropy_renyi(Y: np.ndarray, alpha: float = 2.0, bins: int = 20) -> np.ndarray:
    """Renyi entropy proxy rebroadcast to the original output shape."""
    flat = Y.reshape(len(Y), -1)
    out = np.empty(len(Y))
    for i in range(len(Y)):
        counts, _ = np.histogram(flat[i], bins=bins)
        p = counts / (counts.sum() + 1e-12)
        p = p[p > 0]
        out[i] = (
            -(1.0 / (1.0 - alpha)) * np.log(np.sum(p**alpha) + 1e-30)
            if abs(1.0 - alpha) > 1e-10
            else -np.sum(p * np.log(p + 1e-30))
        )
    return _bc(out, Y) * np.ones_like(Y)
