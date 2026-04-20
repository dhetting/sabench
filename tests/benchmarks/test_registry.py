from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass
from typing import get_args

import pytest

from sabench.benchmarks.base import BenchmarkFunction

REPRESENTATIVE_BENCHMARKS = {
    "Ishigami": "scalar",
    "Campbell2D": "spatial",
    "Lorenz96": "functional",
}


def test_benchmark_types_import() -> None:
    from sabench.benchmarks.types import BenchmarkFamily, BenchmarkSpec

    assert set(get_args(BenchmarkFamily)) == {"scalar", "spatial", "functional"}
    assert is_dataclass(BenchmarkSpec)


def test_registry_exposes_mapping_of_typed_definitions() -> None:
    from sabench.benchmarks.registry import BENCHMARK_REGISTRY, BenchmarkDefinition

    assert isinstance(BENCHMARK_REGISTRY, Mapping)
    assert BENCHMARK_REGISTRY
    assert all(isinstance(item, BenchmarkDefinition) for item in BENCHMARK_REGISTRY.values())


def test_registry_contains_representative_benchmarks_by_family() -> None:
    from sabench.benchmarks.registry import get_benchmark_definition

    for name, family in REPRESENTATIVE_BENCHMARKS.items():
        definition = get_benchmark_definition(name)
        assert definition.spec.name == name
        assert definition.spec.family == family
        assert definition.spec.output_kind == family or (
            family == "spatial" and definition.spec.output_kind == "spatial"
        )
        assert issubclass(definition.benchmark_cls, BenchmarkFunction)


def test_registry_lists_benchmarks_by_family() -> None:
    from sabench.benchmarks.registry import list_benchmarks

    assert "Ishigami" in list_benchmarks("scalar")
    assert "Campbell2D" in list_benchmarks("spatial")
    assert "Lorenz96" in list_benchmarks("functional")


def test_registry_names_are_unique_and_match_specs() -> None:
    from sabench.benchmarks.registry import BENCHMARK_REGISTRY

    names = list(BENCHMARK_REGISTRY)
    assert len(names) == len(set(names))

    for name, definition in BENCHMARK_REGISTRY.items():
        assert definition.spec.name == name
        instance = definition.benchmark_cls()
        assert definition.spec.module.endswith(definition.spec.module_name)
        assert definition.spec.class_name == definition.benchmark_cls.__name__
        assert definition.spec.d == instance.d
        assert len(instance.bounds) == instance.d


def test_benchmark_spec_uses_snake_case_analytical_flags() -> None:
    from dataclasses import fields

    from sabench.benchmarks.types import BenchmarkSpec

    field_names = {field.name for field in fields(BenchmarkSpec)}
    assert "has_analytical_s1" in field_names
    assert "has_analytical_st" in field_names
    assert "has_analytical_S1" not in field_names
    assert "has_analytical_ST" not in field_names


def test_unknown_benchmark_raises_key_error() -> None:
    from sabench.benchmarks.registry import get_benchmark_definition

    with pytest.raises(KeyError, match="Unknown benchmark"):
        get_benchmark_definition("not-a-benchmark")
