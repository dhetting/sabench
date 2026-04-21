"""Layout tests for benchmark test modules."""

from pathlib import Path

import sabench

_PACKAGE_ROOT = Path(sabench.__file__).resolve().parent
_PACKAGE_TEST_ROOT = _PACKAGE_ROOT / "tests"


def test_benchmark_tests_live_at_repo_root() -> None:
    assert (Path("tests") / "benchmarks" / "test_scalar.py").exists()
    assert (Path("tests") / "benchmarks" / "test_spatial.py").exists()
    assert (Path("tests") / "benchmarks" / "test_functional.py").exists()


def test_legacy_packaged_benchmark_tests_removed() -> None:
    assert not (_PACKAGE_TEST_ROOT / "test_scalar.py").exists()
    assert not (_PACKAGE_TEST_ROOT / "test_spatial.py").exists()
    assert not (_PACKAGE_TEST_ROOT / "test_functional.py").exists()
