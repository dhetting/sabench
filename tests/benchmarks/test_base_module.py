from __future__ import annotations

import importlib
from pathlib import Path

import sabench
from sabench.benchmarks.functional import Lorenz96
from sabench.benchmarks.scalar import Ishigami
from sabench.benchmarks.spatial import Campbell2D


def _runtime_root() -> Path:
    return Path(sabench.__file__).resolve().parent


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
    runtime_root = _runtime_root()
    offenders: list[str] = []

    for path in runtime_root.rglob("*.py"):
        relative_path = path.relative_to(runtime_root)
        if relative_path.parts[0] == "tests":
            continue
        if "sabench._base" in path.read_text(encoding="utf-8"):
            offenders.append(str(relative_path))

    assert offenders == []


def test_legacy_base_module_file_removed() -> None:
    assert not (_runtime_root() / "_base.py").exists()
