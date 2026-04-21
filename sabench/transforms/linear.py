"""Linear output transformations."""

from __future__ import annotations

import numpy as np


def t_affine(Y: np.ndarray, a: float = 2.0, b: float = 1.0) -> np.ndarray:
    """Affine pointwise transform: Z(z) = a · Y(z) + b."""
    return a * Y + b
