"""Samplewise output transformations."""

from __future__ import annotations

import numpy as np


def t_temporal_cumsum(Y: np.ndarray) -> np.ndarray:
    """Cumulative sum / running integral: Z(t) = sum_{s<=t} Y(s)."""
    flat = Y.reshape(len(Y), -1)
    return np.cumsum(flat, axis=1).reshape(Y.shape)
