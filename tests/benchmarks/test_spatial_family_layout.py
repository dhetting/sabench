from __future__ import annotations

import importlib
from pathlib import Path

import sabench
from sabench.benchmarks.base import BenchmarkFunction


def _runtime_root() -> Path:
    return Path(sabench.__file__).resolve().parent


def test_spatial_family_imports_from_new_package() -> None:
    module = importlib.import_module("sabench.benchmarks.spatial")

    assert module.Campbell2D.__name__ == "Campbell2D"
    assert module.Campbell3D.__name__ == "Campbell3D"
    assert issubclass(module.Campbell2D, BenchmarkFunction)


def test_runtime_package_does_not_reference_legacy_spatial_package() -> None:
    runtime_root = _runtime_root()
    offenders: list[str] = []

    for path in runtime_root.rglob("*.py"):
        relative_path = path.relative_to(runtime_root)
        if relative_path.parts[0] == "tests":
            continue
        source = path.read_text(encoding="utf-8")
        if "from sabench.spatial import" in source or "import sabench.spatial" in source:
            offenders.append(str(relative_path))

    assert offenders == []


def test_legacy_spatial_package_removed() -> None:
    assert not (_runtime_root() / "spatial").exists()
