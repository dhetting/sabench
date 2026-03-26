"""Tests for the canonical typed benchmark registry."""

from __future__ import annotations

import json
from pathlib import Path

from sabench.benchmarks.scalar import Ishigami
from sabench.benchmarks.spatial import Campbell3D

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = REPO_ROOT / "sabench" / "metadata" / "benchmarks_metadata.json"


def test_typed_benchmark_registry_imports() -> None:
    from sabench.benchmarks.registry import BENCHMARK_SPECS, get_benchmark_class, get_benchmark_spec
    from sabench.benchmarks.types import BenchmarkFamily, BenchmarkSpec, OutputKind

    ishigami_spec = get_benchmark_spec("Ishigami")
    campbell3d_spec = get_benchmark_spec("Campbell3D")

    assert isinstance(ishigami_spec, BenchmarkSpec)
    assert ishigami_spec.family is BenchmarkFamily.SCALAR
    assert ishigami_spec.output_kind is OutputKind.SCALAR
    assert get_benchmark_class("Ishigami") is Ishigami
    assert BENCHMARK_SPECS["Campbell3D"] is campbell3d_spec
    assert campbell3d_spec.family is BenchmarkFamily.SPATIAL
    assert campbell3d_spec.output_kind is OutputKind.SPATIAL
    assert get_benchmark_class("Campbell3D") is Campbell3D
    assert ishigami_spec.module.startswith("sabench.benchmarks.scalar")
    assert campbell3d_spec.module.startswith("sabench.benchmarks.spatial")


def test_benchmark_registry_covers_all_current_benchmark_classes() -> None:
    from sabench.benchmarks.registry import BENCHMARK_SPECS, list_benchmark_names

    assert len(BENCHMARK_SPECS) == 29
    assert list_benchmark_names() == sorted(BENCHMARK_SPECS)


def test_benchmark_registry_keys_match_existing_metadata_snapshot() -> None:
    from sabench.benchmarks.registry import BENCHMARK_SPECS

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert set(BENCHMARK_SPECS) == set(metadata)


def test_benchmark_spec_uses_family_level_output_kind_not_legacy_shape_label() -> None:
    from sabench.benchmarks.registry import get_benchmark_spec
    from sabench.benchmarks.types import OutputKind

    campbell2d_spec = get_benchmark_spec("Campbell2D")
    campbell3d_spec = get_benchmark_spec("Campbell3D")

    assert campbell2d_spec.output_kind is OutputKind.SPATIAL
    assert campbell3d_spec.output_kind is OutputKind.SPATIAL
