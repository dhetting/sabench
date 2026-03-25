"""
Base class for all sabench benchmark functions.

Every benchmark exposes a consistent interface:
  - .evaluate(X)           model evaluations
  - .analytical_S1(...)    first-order indices where closed-form exists
  - .analytical_ST(...)    total-effect indices where closed-form exists
  - .bounds                list of (lo, hi) per input
  - .d                     input dimensionality
  - .output_type           'scalar' | 'spatial' | 'functional'

The mixed-case Sobol method names are intentional. They mirror the notation
used in the sensitivity-analysis literature and are part of the public API.
"""

from __future__ import annotations

import numpy as np


class BenchmarkFunction:
    """Abstract base for all benchmark functions."""

    #: Human-readable name
    name: str = "unnamed"
    #: Number of inputs
    d: int = 0
    #: Input bounds as list of (lo, hi) tuples
    bounds: list = []
    #: One of 'scalar', 'spatial_2d', 'spatial_3d', 'functional'
    output_type: str = "scalar"
    #: Short description for __repr__
    description: str = ""
    #: Literature reference
    reference: str = ""

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the benchmark function.

        Parameters
        ----------
        X : ndarray (n_samples, d)

        Returns
        -------
        Y : ndarray; shape depends on output_type
        """
        raise NotImplementedError

    def __call__(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.evaluate(X, **kwargs)

    def analytical_S1(self, **kwargs) -> np.ndarray | None:
        """First-order Sobol indices.  Returns None if not available."""
        return None

    def analytical_ST(self, **kwargs) -> np.ndarray | None:
        """Total-effect Sobol indices.  Returns None if not available."""
        return None

    def sample(self, N: int, seed: int = None) -> np.ndarray:
        """Draw N uniform samples within bounds (Saltelli A-matrix)."""
        from sabench.sampling.saltelli import saltelli_sample

        return saltelli_sample(self.d, self.bounds, N, seed=seed)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}  d={self.d}  output={self.output_type}  '{self.name}'>"
