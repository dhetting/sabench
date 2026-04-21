from __future__ import annotations

from pathlib import Path

import numpy as np

import sabench
from sabench.transforms.utilities import _bc, _safe_range, _ymin


def test_transform_utilities_module_exists() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    assert (package_root / "transforms" / "utilities.py").exists()


def test_transform_utilities_helpers_match_expected_behavior() -> None:
    y = np.array(
        [
            [[1.0, 3.0], [2.0, 5.0]],
            [[-2.0, 4.0], [0.0, 1.0]],
        ]
    )

    mins = _ymin(y)
    ranges = _safe_range(y)
    broadcast = _bc(mins, y)

    np.testing.assert_allclose(mins, np.array([1.0, -2.0]))
    np.testing.assert_allclose(ranges, np.array([4.0, 6.0]))
    np.testing.assert_allclose(broadcast[:, 0, 0], mins)
    assert broadcast.shape == (2, 1, 1)


def test_transform_monolith_no_longer_defines_shared_helpers() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    transforms_source = (package_root / "transforms" / "transforms.py").read_text()

    assert "def _safe_range(" not in transforms_source
    assert "def _ymin(" not in transforms_source
    assert "def _bc(" not in transforms_source
