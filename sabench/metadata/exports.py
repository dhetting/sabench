"""Canonical metadata exports generated from typed registries."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sabench.benchmarks import get_benchmark_spec, list_benchmarks
from sabench.transforms import get_transform_spec, list_transforms

BENCHMARKS_REGISTRY_EXPORT_FILENAME = "benchmarks_registry_metadata.json"
TRANSFORMS_REGISTRY_EXPORT_FILENAME = "transforms_registry_metadata.json"


def _json_compatible(value: Any) -> Any:
    """Convert nested registry metadata into JSON-compatible builtins."""
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    return value


def export_registered_benchmark_metadata() -> dict[str, dict[str, Any]]:
    """Return canonical benchmark metadata for the typed benchmark registry."""
    return {
        name: _json_compatible(asdict(get_benchmark_spec(name)))
        for name in list_benchmarks()
    }


def export_registered_transform_metadata() -> dict[str, dict[str, Any]]:
    """Return canonical transform metadata for the typed transform registry."""
    return {
        key: _json_compatible(asdict(get_transform_spec(key)))
        for key in list_transforms()
    }


def write_registry_metadata_exports(metadata_root: Path | None = None) -> tuple[Path, Path]:
    """Write canonical registry metadata export snapshots to disk."""
    if metadata_root is None:
        metadata_root = Path(__file__).resolve().parent

    metadata_root.mkdir(parents=True, exist_ok=True)

    benchmarks_path = metadata_root / BENCHMARKS_REGISTRY_EXPORT_FILENAME
    transforms_path = metadata_root / TRANSFORMS_REGISTRY_EXPORT_FILENAME

    benchmarks_path.write_text(
        json.dumps(export_registered_benchmark_metadata(), indent=2, sort_keys=True) + "\n"
    )
    transforms_path.write_text(
        json.dumps(export_registered_transform_metadata(), indent=2, sort_keys=True) + "\n"
    )

    return benchmarks_path, transforms_path
