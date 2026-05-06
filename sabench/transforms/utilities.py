"""Shared internal helpers for transform implementations."""

from __future__ import annotations

import numpy as np


def _safe_range(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return per-sample ranges with a lower numerical floor.

    Handles empty inputs and 0-/1-D per-sample inputs safely.
    """
    y = np.asarray(y)
    # Empty input -> return empty 1-D array
    if y.size == 0:
        return np.zeros(0, dtype=float)
    # Treat 0-D scalar as single sample
    if y.ndim == 0:
        flat = y.reshape(1, 1)
    else:
        flat = y.reshape(len(y), -1)
    # If there are no non-sample dims, return eps per sample
    if flat.shape[1] == 0:
        return np.full(flat.shape[0], eps, dtype=float)
    rng = flat.max(axis=1) - flat.min(axis=1)
    return np.maximum(rng, eps)


def _ymin(y: np.ndarray) -> np.ndarray:
    """Return the per-sample minimum across all non-sample dimensions.

    Empty inputs return an empty array.
    """
    y = np.asarray(y)
    if y.size == 0:
        return np.zeros(0, dtype=float)
    if y.ndim == 0:
        flat = y.reshape(1, 1)
    else:
        flat = y.reshape(len(y), -1)
    if flat.shape[1] == 0:
        return np.full(flat.shape[0], np.nan, dtype=float)
    return flat.min(axis=1)


def _bc(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Broadcast a per-sample vector to the shape of a batched output array.

    Accepts:
    - v scalar -> returns an array shaped (n, 1, ..., 1) filled with the scalar.
    - v 1-D of length n -> returns v reshaped to (n, 1, ..., 1).
    Raises ValueError for length mismatches.
    """
    y = np.asarray(y)
    v_arr = np.asarray(v)

    # If y has no batch dimension (0-D), attempt to return scalar or raise
    if y.ndim == 0:
        if v_arr.ndim == 0:
            return v_arr
        if v_arr.ndim == 1 and v_arr.size == 1:
            return v_arr[0]
        raise ValueError("y has no batch dimension; cannot broadcast v")

    n = y.shape[0]

    if v_arr.ndim == 0:
        return np.full((n,) + (1,) * (y.ndim - 1), v_arr, dtype=v_arr.dtype)
    if v_arr.ndim == 1:
        if v_arr.shape[0] != n:
            raise ValueError(
                f"v length ({v_arr.shape[0]}) does not match number of samples in y ({n})"
            )
        return v_arr.reshape((n,) + (1,) * (y.ndim - 1))
    # For higher-d inputs, require leading dimension to match n
    if v_arr.shape[0] != n:
        raise ValueError(
            f"First dimension of v ({v_arr.shape[0]}) does not match number of samples in y ({n})"
        )
    return v_arr.reshape((n,) + (1,) * (y.ndim - 1))
