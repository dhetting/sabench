"""Benchmark package contracts and registries."""

from sabench.benchmarks.base import BenchmarkFunction
from sabench.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BenchmarkDefinition,
    get_benchmark_definition,
    get_benchmark_spec,
    list_benchmarks,
)
from sabench.benchmarks.types import BenchmarkFamily, BenchmarkOutputKind, BenchmarkSpec

__all__ = [
    "BENCHMARK_REGISTRY",
    "BenchmarkDefinition",
    "BenchmarkFamily",
    "BenchmarkFunction",
    "BenchmarkOutputKind",
    "BenchmarkSpec",
    "get_benchmark_definition",
    "get_benchmark_spec",
    "list_benchmarks",
]
