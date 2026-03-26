"""Architecture tests for the Slice 7 transform module split."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_TRANSFORM_MODULE = REPO_ROOT / "sabench" / "transforms" / "transforms.py"


def test_representative_split_transform_specs_use_new_modules() -> None:
    from sabench.transforms.registry import get_transform_spec

    assert get_transform_spec("affine_a2_b1").module == "sabench.transforms.pointwise"
    assert get_transform_spec("tanh_a03").module == "sabench.transforms.pointwise"
    assert get_transform_spec("softplus_b01").module == "sabench.transforms.pointwise"
    assert get_transform_spec("temporal_cumsum").module == "sabench.transforms.linear"
    assert get_transform_spec("temporal_bandpass").module == "sabench.transforms.linear"


def test_legacy_monolith_no_longer_defines_moved_representative_functions() -> None:
    text = LEGACY_TRANSFORM_MODULE.read_text(encoding="utf-8")

    forbidden_defs = [
        "def t_affine(",
        "def t_tanh_pointwise(",
        "def t_softplus_pointwise(",
        "def t_temporal_cumsum(",
        "def t_temporal_bandpass(",
    ]

    offenders = [marker for marker in forbidden_defs if marker in text]
    assert offenders == []
