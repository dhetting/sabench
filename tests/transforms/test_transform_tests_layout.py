from __future__ import annotations

from pathlib import Path

import sabench


def test_packaged_transform_tests_are_removed() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    assert not (package_root / "tests" / "test_transforms.py").exists()
    assert not (package_root / "tests" / "test_transforms_expanded.py").exists()
