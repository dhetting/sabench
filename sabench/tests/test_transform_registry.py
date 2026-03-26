"""Tests for the typed transform registry skeleton."""

from __future__ import annotations

import json
from pathlib import Path

from sabench.benchmarks.types import OutputKind
from sabench.transforms.transforms import TRANSFORMS as LEGACY_TRANSFORMS

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = REPO_ROOT / "sabench" / "metadata" / "transforms_metadata.json"


def test_typed_transform_registry_imports() -> None:
    from sabench.transforms.base import TransformFunction
    from sabench.transforms.registry import TRANSFORM_SPECS, get_transform, get_transform_spec
    from sabench.transforms.types import TransformCategory, TransformSpec, TransformTag

    affine_spec = get_transform_spec("affine_a2_b1")
    regional_mean_spec = get_transform_spec("regional_mean")
    temporal_cumsum_spec = get_transform_spec("temporal_cumsum")

    assert isinstance(affine_spec, TransformSpec)
    assert isinstance(affine_spec.transform_fn, TransformFunction)
    assert affine_spec.category is TransformCategory.ENGINEERING
    assert TransformTag.AFFINE in affine_spec.tags
    assert TransformTag.LINEAR in affine_spec.tags
    assert TransformTag.POINTWISE in affine_spec.tags
    assert get_transform("affine_a2_b1") is affine_spec.transform_fn
    assert TRANSFORM_SPECS["regional_mean"] is regional_mean_spec
    assert regional_mean_spec.category is TransformCategory.SPATIAL
    assert TransformTag.NONLOCAL in regional_mean_spec.tags
    assert temporal_cumsum_spec.category is TransformCategory.TEMPORAL
    assert TransformTag.LINEAR in temporal_cumsum_spec.tags
    assert TransformTag.NONLOCAL in temporal_cumsum_spec.tags


def test_transform_registry_covers_existing_transform_keys() -> None:
    from sabench.transforms.registry import TRANSFORM_SPECS, list_transform_names

    assert set(TRANSFORM_SPECS) == set(LEGACY_TRANSFORMS)
    assert list_transform_names() == sorted(TRANSFORM_SPECS)


def test_transform_registry_matches_existing_metadata_snapshot_keys() -> None:
    from sabench.transforms.registry import TRANSFORM_SPECS

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert set(TRANSFORM_SPECS) == set(metadata)


def test_transform_specs_have_valid_category_tags_and_output_kinds() -> None:
    from sabench.transforms.registry import TRANSFORM_SPECS
    from sabench.transforms.types import TransformCategory, TransformSpec, TransformTag

    valid_categories = set(TransformCategory)
    valid_tags = set(TransformTag)

    for spec in TRANSFORM_SPECS.values():
        assert isinstance(spec, TransformSpec)
        assert spec.category in valid_categories
        assert callable(spec.transform_fn)
        assert spec.supported_output_kinds
        assert set(spec.supported_output_kinds) <= set(OutputKind)
        assert set(spec.tags) <= valid_tags
