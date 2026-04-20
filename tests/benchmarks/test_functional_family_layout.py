from __future__ import annotations

import importlib
from pathlib import Path

import sabench
from sabench.benchmarks.base import BenchmarkFunction


def _runtime_root() -> Path:
    return Path(sabench.__file__).resolve().parent


def test_functional_family_imports_from_new_package() -> None:
    module = importlib.import_module("sabench.benchmarks.functional")

    assert module.Lorenz96.__name__ == "Lorenz96"
    assert module.DampedOscillator.__name__ == "DampedOscillator"
    assert issubclass(module.Lorenz96, BenchmarkFunction)


def test_runtime_package_does_not_reference_legacy_functional_package() -> None:
    runtime_root = _runtime_root()
    offenders: list[str] = []

    for path in runtime_root.rglob("*.py"):
        relative_path = path.relative_to(runtime_root)
        if relative_path.parts[0] == "tests":
            continue
        source = path.read_text(encoding="utf-8")
        if "from sabench.functional import" in source or "import sabench.functional" in source:
            offenders.append(str(relative_path))

    assert offenders == []


def test_legacy_functional_package_removed() -> None:
    assert not (_runtime_root() / "functional").exists()
