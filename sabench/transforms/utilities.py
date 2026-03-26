"""Shared helpers for modular transform implementations."""

from __future__ import annotations

import numpy as np


def flatten_samples(Y: np.ndarray) -> np.ndarray:
    """Flatten each sample to a 1D feature axis while preserving sample count."""

    array = np.asarray(Y)
    return array.reshape(len(array), -1)
