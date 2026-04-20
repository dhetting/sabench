"""Canonical typed benchmark registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sabench.benchmarks.base import BenchmarkFunction
from sabench.benchmarks.scalar import Ishigami
from sabench.benchmarks.spatial import Campbell2D
from sabench.benchmarks.types import BenchmarkFamily, BenchmarkOutputKind, BenchmarkSpec
from sabench.functional import Lorenz96


@dataclass(frozen=True, slots=True)
class BenchmarkDefinition:
    """Canonical registry entry for a benchmark."""

    spec: BenchmarkSpec
    benchmark_cls: type[BenchmarkFunction]


def _has_overridden_method(benchmark_cls: type[BenchmarkFunction], method_name: str) -> bool:
    """Return whether a benchmark overrides an analytical API method."""
    return getattr(benchmark_cls, method_name) is not getattr(BenchmarkFunction, method_name)


def _build_spec(
    benchmark_cls: type[BenchmarkFunction],
    *,
    family: BenchmarkFamily,
    output_kind: BenchmarkOutputKind,
    module_name: str,
) -> BenchmarkSpec:
    """Build canonical typed metadata directly from the benchmark class."""
    instance = benchmark_cls()
    return BenchmarkSpec(
        name=benchmark_cls.name,
        family=family,
        module=benchmark_cls.__module__,
        module_name=module_name,
        class_name=benchmark_cls.__name__,
        output_kind=output_kind,
        d=instance.d,
        description=benchmark_cls.description,
        reference=benchmark_cls.reference,
        has_analytical_s1=_has_overridden_method(benchmark_cls, "analytical_S1"),
        has_analytical_st=_has_overridden_method(benchmark_cls, "analytical_ST"),
    )


_REGISTRY: dict[str, BenchmarkDefinition] = {
    "Ishigami": BenchmarkDefinition(
        spec=_build_spec(
            Ishigami,
            family="scalar",
            output_kind="scalar",
            module_name="ishigami",
        ),
        benchmark_cls=Ishigami,
    ),
    "Campbell2D": BenchmarkDefinition(
        spec=_build_spec(
            Campbell2D,
            family="spatial",
            output_kind="spatial",
            module_name="campbell2d",
        ),
        benchmark_cls=Campbell2D,
    ),
    "Lorenz96": BenchmarkDefinition(
        spec=_build_spec(
            Lorenz96,
            family="functional",
            output_kind="functional",
            module_name="lorenz96",
        ),
        benchmark_cls=Lorenz96,
    ),
}

BENCHMARK_REGISTRY: Mapping[str, BenchmarkDefinition] = MappingProxyType(_REGISTRY)


def get_benchmark_definition(name: str) -> BenchmarkDefinition:
    """Return the registry entry for a benchmark name."""
    try:
        return BENCHMARK_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark: {name}") from exc


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    """Return the typed spec for a benchmark name."""
    return get_benchmark_definition(name).spec


def list_benchmarks(family: BenchmarkFamily | None = None) -> tuple[str, ...]:
    """List registered benchmark names, optionally filtered by family."""
    names = [
        name
        for name, definition in BENCHMARK_REGISTRY.items()
        if family is None or definition.spec.family == family
    ]
    return tuple(sorted(names))
