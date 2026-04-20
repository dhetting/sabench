from __future__ import annotations

import importlib
from pathlib import Path

from sabench.benchmarks.base import BenchmarkFunction


def test_scalar_family_imports_from_new_package() -> None:
    module = importlib.import_module("sabench.benchmarks.scalar")

    assert module.Ishigami.__name__ == "Ishigami"
    assert module.Borehole.__name__ == "Borehole"
    assert issubclass(module.Ishigami, BenchmarkFunction)


def test_runtime_package_does_not_reference_legacy_scalar_package() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    runtime_root = repo_root / "sabench"
    offenders: list[str] = []

    for path in runtime_root.rglob("*.py"):
        relative_path = path.relative_to(runtime_root)
        if relative_path.parts[0] == "tests":
            continue
        source = path.read_text(encoding="utf-8")
        if "from sabench.scalar import" in source or "import sabench.scalar" in source:
            offenders.append(str(relative_path))

    assert offenders == []


def test_legacy_scalar_package_removed() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "sabench" / "scalar").exists()
