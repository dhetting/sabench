"""Canonical typed benchmark registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from sabench.benchmarks import functional, scalar, spatial
from sabench.benchmarks.base import BenchmarkFunction
from sabench.benchmarks.types import BenchmarkFamily, BenchmarkOutputKind, BenchmarkSpec


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
) -> BenchmarkSpec:
    """Build canonical typed metadata directly from the benchmark class."""
    instance = benchmark_cls()
    return BenchmarkSpec(
        name=benchmark_cls.name,
        family=family,
        module=benchmark_cls.__module__,
        module_name=benchmark_cls.__module__.rsplit(".", maxsplit=1)[-1],
        class_name=benchmark_cls.__name__,
        output_kind=output_kind,
        d=instance.d,
        description=benchmark_cls.description,
        reference=benchmark_cls.reference,
        has_analytical_s1=_has_overridden_method(benchmark_cls, "analytical_S1"),
        has_analytical_st=_has_overridden_method(benchmark_cls, "analytical_ST"),
    )


def _build_definition(
    benchmark_cls: type[BenchmarkFunction],
    *,
    family: BenchmarkFamily,
    output_kind: BenchmarkOutputKind,
) -> BenchmarkDefinition:
    """Build a canonical benchmark definition for a benchmark class."""
    return BenchmarkDefinition(
        spec=_build_spec(
            benchmark_cls,
            family=family,
            output_kind=output_kind,
        ),
        benchmark_cls=benchmark_cls,
    )


def _build_family_registry(
    package_module: Any,
    *,
    family: BenchmarkFamily,
    output_kind: BenchmarkOutputKind,
) -> dict[str, BenchmarkDefinition]:
    """Build canonical registry entries for a benchmark family package."""
    names = cast(list[str], package_module.__all__)
    return {
        name: _build_definition(
            cast(type[BenchmarkFunction], getattr(package_module, name)),
            family=family,
            output_kind=output_kind,
        )
        for name in names
    }


_REGISTRY: dict[str, BenchmarkDefinition] = {}
_REGISTRY.update(_build_family_registry(scalar, family="scalar", output_kind="scalar"))
_REGISTRY.update(_build_family_registry(spatial, family="spatial", output_kind="spatial"))
_REGISTRY.update(_build_family_registry(functional, family="functional", output_kind="functional"))

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
