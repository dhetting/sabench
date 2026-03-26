"""Canonical package anchor for benchmark definitions."""

from sabench.benchmarks.base import BenchmarkFunction
from sabench.benchmarks.registry import (
    BENCHMARK_SPECS,
    get_benchmark_class,
    get_benchmark_spec,
    list_benchmark_names,
)
from sabench.benchmarks.types import BenchmarkFamily, BenchmarkSpec, OutputKind

__all__ = [
    "BenchmarkFamily",
    "BenchmarkFunction",
    "BenchmarkSpec",
    "BENCHMARK_SPECS",
    "OutputKind",
    "get_benchmark_class",
    "get_benchmark_spec",
    "list_benchmark_names",
]
