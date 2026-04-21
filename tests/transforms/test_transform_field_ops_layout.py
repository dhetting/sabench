from __future__ import annotations

from pathlib import Path

import sabench
from sabench.transforms import TRANSFORMS, get_transform_spec
from sabench.transforms.field_ops import t_gradient_magnitude


def test_gradient_magnitude_spec_points_to_field_ops_module() -> None:
    spec = get_transform_spec("gradient_magnitude")
    assert spec.mechanism == "field_op"
    assert spec.module == "sabench.transforms.field_ops"


def test_legacy_registry_uses_field_ops_function() -> None:
    assert TRANSFORMS["gradient_magnitude"]["fn"] is t_gradient_magnitude


def test_field_ops_module_exists() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    assert (package_root / "transforms" / "field_ops.py").exists()
