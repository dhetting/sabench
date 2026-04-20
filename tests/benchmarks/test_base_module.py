from __future__ import annotations

import importlib
from pathlib import Path

from sabench.functional import Lorenz96
from sabench.scalar import Ishigami
from sabench.spatial import Campbell2D


def test_new_base_module_import_succeeds() -> None:
    module = importlib.import_module("sabench.benchmarks.base")
    from sabench.benchmarks.base import BenchmarkFunction

    assert module.BenchmarkFunction is BenchmarkFunction


def test_representative_benchmarks_inherit_from_new_base_module() -> None:
    from sabench.benchmarks.base import BenchmarkFunction

    assert issubclass(Ishigami, BenchmarkFunction)
    assert issubclass(Campbell2D, BenchmarkFunction)
    assert issubclass(Lorenz96, BenchmarkFunction)


def test_runtime_package_does_not_reference_old_base_module() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    runtime_root = repo_root / "sabench"
    offenders: list[str] = []

    for path in runtime_root.rglob("*.py"):
        relative_path = path.relative_to(runtime_root)
        if relative_path.parts[0] == "tests":
            continue
        if "sabench._base" in path.read_text(encoding="utf-8"):
            offenders.append(str(relative_path))

    assert offenders == []


def test_legacy_base_module_file_removed() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "sabench" / "_base.py").exists()
