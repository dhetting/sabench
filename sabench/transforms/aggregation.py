"""Aggregation transforms extracted from the legacy transform monolith."""

from __future__ import annotations

import numpy as np


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
