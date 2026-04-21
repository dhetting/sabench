"""Base callable contracts for transforms."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class TransformFunction(Protocol):
    """Callable transform function with inspectable function metadata."""

    __name__: str
    __module__: str

    def __call__(self, y: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray: ...


class BoundTransform(Protocol):
    """Callable transform with parameters already bound."""

    def __call__(self, y: np.ndarray) -> np.ndarray: ...
