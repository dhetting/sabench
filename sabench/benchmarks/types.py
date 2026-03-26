"""Typed benchmark metadata definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sabench.benchmarks.base import BenchmarkFunction


class OutputKind(str, Enum):
    """Primary scientific output families exposed by sabench benchmarks."""

    SCALAR = "scalar"
    SPATIAL = "spatial"
    FUNCTIONAL = "functional"


class BenchmarkFamily(str, Enum):
    """Filesystem grouping for benchmark implementations."""

    SCALAR = "scalar"
    SPATIAL = "spatial"
    FUNCTIONAL = "functional"


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Canonical typed definition for a benchmark entry in the registry."""

    key: str
    benchmark_cls: type[BenchmarkFunction]
    family: BenchmarkFamily
    output_kind: OutputKind
    module: str
