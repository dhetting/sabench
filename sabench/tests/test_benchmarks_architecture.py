"""Architecture tests for the new benchmark package anchor."""

from __future__ import annotations

from pathlib import Path

from sabench.benchmarks.functional import Lorenz96
from sabench.benchmarks.scalar import Ishigami
from sabench.benchmarks.spatial import Campbell2D

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PACKAGE = REPO_ROOT / "sabench"


def test_benchmark_function_imports_from_new_package_anchor() -> None:
    from sabench.benchmarks.base import BenchmarkFunction

    assert BenchmarkFunction.__module__ == "sabench.benchmarks.base"


def test_representative_benchmark_families_use_new_base_contract() -> None:
    from sabench.benchmarks.base import BenchmarkFunction

    assert issubclass(Ishigami, BenchmarkFunction)
    assert issubclass(Campbell2D, BenchmarkFunction)
    assert issubclass(Lorenz96, BenchmarkFunction)


def test_runtime_package_has_no_legacy_base_module() -> None:
    assert not (RUNTIME_PACKAGE / "_base.py").exists()


def test_runtime_python_files_do_not_import_legacy_base_module() -> None:
    offenders: list[str] = []
    for path in RUNTIME_PACKAGE.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "sabench._base" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_runtime_package_has_no_legacy_scalar_family_package() -> None:
    assert not (RUNTIME_PACKAGE / "scalar").exists()


def test_runtime_python_files_do_not_import_legacy_scalar_family_package() -> None:
    offenders: list[str] = []
    for path in RUNTIME_PACKAGE.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "sabench.scalar" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_runtime_package_has_no_legacy_spatial_family_package() -> None:
    assert not (RUNTIME_PACKAGE / "spatial").exists()


def test_runtime_python_files_do_not_import_legacy_spatial_family_package() -> None:
    offenders: list[str] = []
    for path in RUNTIME_PACKAGE.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "sabench.spatial" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_runtime_package_has_no_legacy_functional_family_package() -> None:
    assert not (RUNTIME_PACKAGE / "functional").exists()


def test_runtime_python_files_do_not_import_legacy_functional_family_package() -> None:
    offenders: list[str] = []
    for path in RUNTIME_PACKAGE.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "sabench.functional" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
