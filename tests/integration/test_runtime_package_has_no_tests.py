from pathlib import Path

import sabench


def test_runtime_package_has_no_tests_directory() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    assert not (package_root / "tests").exists()
