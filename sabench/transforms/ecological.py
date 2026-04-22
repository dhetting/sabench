"""Ecological and compositional transforms."""

from __future__ import annotations

import numpy as np

from sabench.transforms.utilities import _bc, _ymin


def t_hellinger(Y: np.ndarray) -> np.ndarray:
    """Hellinger transform: phi(u) = sqrt(u/sum(u)) -- chi^2 distance equaliser."""
    flat = np.maximum(Y.reshape(len(Y), -1), 0.0)
    row_sum = flat.sum(axis=1, keepdims=True) + 1e-12
    return np.sqrt(flat / row_sum).reshape(Y.shape)


def t_chord_dist(Y: np.ndarray) -> np.ndarray:
    """Chord normalisation: phi(u) = u/||u|| -- L2 unit sphere projection."""
    flat = Y.reshape(len(Y), -1)
    norm = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    return (flat / norm).reshape(Y.shape)


def t_relative_abundance(Y: np.ndarray) -> np.ndarray:
    """Relative abundance: phi(u) = u/sum(u) -- simplex projection, sums to 1."""
    flat = np.maximum(Y.reshape(len(Y), -1), 0.0)
    row_sum = flat.sum(axis=1, keepdims=True) + 1e-12
    return (flat / row_sum).reshape(Y.shape)


def t_log_ratio(Y: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Log-ratio (isometric log-ratio like): phi(y) = log(y - ymin + eps) - mean_log."""
    s = _bc(_ymin(Y), Y)
    log_y = np.log(Y - s + eps)
    mu = _bc(log_y.reshape(len(Y), -1).mean(axis=1), Y)
    return log_y - mu
