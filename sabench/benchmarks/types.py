"""Typed benchmark metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BenchmarkFamily = Literal["scalar", "spatial", "functional"]
BenchmarkOutputKind = Literal["scalar", "spatial", "functional"]


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Canonical typed metadata for a benchmark definition."""

    name: str
    family: BenchmarkFamily
    module: str
    module_name: str
    class_name: str
    output_kind: BenchmarkOutputKind
    d: int
    description: str
    reference: str
    has_analytical_s1: bool
    has_analytical_st: bool
