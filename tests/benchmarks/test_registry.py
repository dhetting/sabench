from __future__ import annotations

import json
from pathlib import Path

import sabench
from sabench.benchmarks import BENCHMARK_REGISTRY, get_benchmark_spec, list_benchmarks
from sabench.benchmarks.functional import __all__ as functional_all
from sabench.benchmarks.scalar import __all__ as scalar_all
from sabench.benchmarks.spatial import __all__ as spatial_all


def test_representative_benchmark_specs_point_to_new_modules() -> None:
    assert get_benchmark_spec("Ishigami").module == "sabench.benchmarks.scalar.ishigami"
    assert get_benchmark_spec("Campbell2D").module == "sabench.benchmarks.spatial.campbell2d"
    assert get_benchmark_spec("Lorenz96").module == "sabench.benchmarks.functional.lorenz96"


def test_benchmark_registry_definition_class_matches_spec_name() -> None:
    for name, definition in BENCHMARK_REGISTRY.items():
        assert definition.spec.class_name == name
        assert definition.benchmark_cls.__name__ == definition.spec.class_name


def test_benchmark_registry_covers_all_benchmark_families() -> None:
    assert set(list_benchmarks("scalar")) == set(scalar_all)
    assert set(list_benchmarks("spatial")) == set(spatial_all)
    assert set(list_benchmarks("functional")) == set(functional_all)
    assert set(list_benchmarks()) == set(scalar_all) | set(spatial_all) | set(functional_all)


def test_benchmark_registry_family_counts_match_package_catalogue() -> None:
    assert len(list_benchmarks("scalar")) == 19
    assert len(list_benchmarks("spatial")) == 3
    assert len(list_benchmarks("functional")) == 7
    assert len(list_benchmarks()) == 29


def test_benchmark_registry_export_snapshot_covers_full_catalogue() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    snapshot_path = package_root / "metadata" / "benchmarks_registry_metadata.json"
    data = json.loads(snapshot_path.read_text())

    assert set(data) == set(list_benchmarks())
    assert len(data) == 29


def test_benchmark_spec_uses_snake_case_analytical_flags() -> None:
    fields = set(get_benchmark_spec("Ishigami").__dataclass_fields__)
    assert "has_analytical_s1" in fields
    assert "has_analytical_st" in fields
    assert "has_analytical_S1" not in fields
    assert "has_analytical_ST" not in fields
