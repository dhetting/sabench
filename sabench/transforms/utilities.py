"""Shared internal helpers for transform implementations."""

from __future__ import annotations

import numpy as np


def _safe_range(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return per-sample ranges with a lower numerical floor."""
    flat = y.reshape(len(y), -1)
    return (flat.max(axis=1) - flat.min(axis=1)).clip(min=eps)


def _ymin(y: np.ndarray) -> np.ndarray:
    """Return the per-sample minimum across all non-sample dimensions."""
    return y.reshape(len(y), -1).min(axis=1)


def _bc(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Broadcast a per-sample vector to the shape of a batched output array."""
    return v.reshape((len(v),) + (1,) * (y.ndim - 1))
