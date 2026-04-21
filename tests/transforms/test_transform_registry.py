from __future__ import annotations

import numpy as np
import pytest

from sabench.transforms import (
    TRANSFORM_REGISTRY,
    TransformDefinition,
    TransformSpec,
    get_transform,
    get_transform_definition,
    get_transform_spec,
    list_transforms,
)

EXPECTED_REGISTRY_KEYS = (
    "affine_a2_b1",
    "tanh_a03",
    "softplus_b01",
    "temporal_cumsum",
    "temporal_peak",
    "gradient_magnitude",
)
EXPECTED_LIST_KEYS = tuple(sorted(EXPECTED_REGISTRY_KEYS))
VALID_OUTPUT_KINDS = {"scalar", "spatial", "functional"}
VALID_MECHANISMS = {"pointwise", "samplewise", "aggregation", "field_op"}
VALID_TAGS = {
    "affine",
    "linear",
    "nonlinear",
    "pointwise",
    "nonlocal",
    "convex",
    "concave",
    "monotone",
    "nonmonotone",
    "smooth",
    "nonsmooth",
    "environmental",
    "engineering",
    "spatial",
    "temporal",
    "statistical",
    "mathematical",
}


def test_transform_registry_exposes_expected_subset() -> None:
    assert list_transforms() == EXPECTED_LIST_KEYS
    assert tuple(TRANSFORM_REGISTRY.keys()) == EXPECTED_REGISTRY_KEYS


def test_transform_registry_entries_are_typed_and_callable() -> None:
    y = np.linspace(-2.0, 2.0, 12, dtype=float).reshape(3, 4)

    for key in EXPECTED_REGISTRY_KEYS:
        definition = get_transform_definition(key)
        assert isinstance(definition, TransformDefinition)
        assert isinstance(definition.spec, TransformSpec)
        assert callable(definition.transform)

        direct = definition.transform(y)
        via_lookup = get_transform(key)(y)
        np.testing.assert_allclose(direct, via_lookup)


def test_transform_spec_fields_are_valid() -> None:
    for key in EXPECTED_REGISTRY_KEYS:
        spec = get_transform_spec(key)
        assert spec.key == key
        assert spec.mechanism in VALID_MECHANISMS
        assert spec.supported_output_kinds
        assert set(spec.supported_output_kinds) <= VALID_OUTPUT_KINDS
        assert set(spec.tags) <= VALID_TAGS
        expected_module = {
            "affine_a2_b1": "sabench.transforms.pointwise",
            "tanh_a03": "sabench.transforms.pointwise",
            "softplus_b01": "sabench.transforms.nonlinear",
            "temporal_cumsum": "sabench.transforms.samplewise",
            "temporal_peak": "sabench.transforms.aggregation",
            "gradient_magnitude": "sabench.transforms.field_ops",
        }[key]
        assert spec.module == expected_module
        assert spec.function_name.startswith("t_")
        assert spec.reference


def test_transform_registry_can_filter_by_mechanism() -> None:
    assert list_transforms(mechanism="pointwise") == ("affine_a2_b1", "softplus_b01", "tanh_a03")
    assert list_transforms(mechanism="samplewise") == ("temporal_cumsum",)
    assert list_transforms(mechanism="aggregation") == ("temporal_peak",)
    assert list_transforms(mechanism="field_op") == ("gradient_magnitude",)


def test_unknown_transform_lookup_raises_clear_error() -> None:
    with pytest.raises(KeyError, match="Unknown transform"):
        get_transform_definition("does_not_exist")
