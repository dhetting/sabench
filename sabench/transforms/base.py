"""Base callable contract for sabench transforms."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TransformFunction(Protocol):
    """Callable protocol for transforms operating on benchmark outputs."""

    def __call__(self, Y: np.ndarray, /, **kwargs: object) -> np.ndarray: ...
