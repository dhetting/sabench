"""Field-operation transforms extracted from the legacy transform monolith."""

from __future__ import annotations

import numpy as np


def t_gradient_magnitude(Y: np.ndarray) -> np.ndarray:
    """Per-sample spatial gradient magnitude field |∇Y|."""
    if Y.ndim < 3:
        return np.zeros_like(Y)
    y_out = np.empty_like(Y, dtype=float)
    for sample_idx in range(len(Y)):
        gradients = np.gradient(Y[sample_idx].astype(float))
        y_out[sample_idx] = np.sqrt(sum(gradient**2 for gradient in gradients))
    return y_out
