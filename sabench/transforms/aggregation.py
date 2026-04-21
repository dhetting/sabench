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
