from __future__ import annotations

import json
from pathlib import Path

import sabench

from sabench.metadata.exports import (
    BENCHMARKS_REGISTRY_EXPORT_FILENAME,
    TRANSFORMS_REGISTRY_EXPORT_FILENAME,
    export_registered_benchmark_metadata,
    export_registered_transform_metadata,
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def test_registry_metadata_export_files_exist() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    metadata_root = package_root / "metadata"

    assert (metadata_root / BENCHMARKS_REGISTRY_EXPORT_FILENAME).exists()
    assert (metadata_root / TRANSFORMS_REGISTRY_EXPORT_FILENAME).exists()


def test_benchmark_registry_export_file_matches_canonical_export() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    metadata_root = package_root / "metadata"

    exported = export_registered_benchmark_metadata()
    on_disk = _load_json(metadata_root / BENCHMARKS_REGISTRY_EXPORT_FILENAME)

    assert on_disk == exported


def test_transform_registry_export_file_matches_canonical_export() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    metadata_root = package_root / "metadata"

    exported = export_registered_transform_metadata()
    on_disk = _load_json(metadata_root / TRANSFORMS_REGISTRY_EXPORT_FILENAME)

    assert on_disk == exported


def test_benchmark_registry_export_uses_canonical_fields() -> None:
    exported = export_registered_benchmark_metadata()

    assert exported["Ishigami"]["module"] == "sabench.benchmarks.scalar.ishigami"
    assert exported["Ishigami"]["output_kind"] == "scalar"
    assert exported["Campbell2D"]["family"] == "spatial"
    assert exported["Lorenz96"]["has_analytical_st"] is False


def test_transform_registry_export_uses_canonical_fields() -> None:
    exported = export_registered_transform_metadata()

    assert exported["affine_a2_b1"]["module"] == "sabench.transforms.linear"
    assert exported["cube_pointwise"]["module"] == "sabench.transforms.pointwise"
    assert exported["sinc"]["module"] == "sabench.transforms.pointwise"
    assert exported["square_wave"]["module"] == "sabench.transforms.pointwise"
    assert exported["softplus_b01"]["module"] == "sabench.transforms.nonlinear"
    assert exported["cosh_pointwise"]["module"] == "sabench.transforms.nonlinear"
    assert exported["logistic_pointwise"]["module"] == "sabench.transforms.nonlinear"
    assert exported["soft_threshold"]["module"] == "sabench.transforms.nonlinear"
    assert exported["hard_threshold"]["module"] == "sabench.transforms.nonlinear"
    assert exported["ramp"]["module"] == "sabench.transforms.nonlinear"
    assert exported["spike_gaussian"]["module"] == "sabench.transforms.nonlinear"
    assert exported["breakpoint"]["module"] == "sabench.transforms.nonlinear"
    assert exported["hockey_stick"]["module"] == "sabench.transforms.nonlinear"
    assert exported["deadzone"]["module"] == "sabench.transforms.nonlinear"
    assert exported["temporal_cumsum"]["mechanism"] == "samplewise"
    assert exported["gradient_magnitude"]["supported_output_kinds"] == ["spatial"]
