from __future__ import annotations

import numpy as np
import pytest

from sabench.transforms import (
    AFFINE_TRANSFORMS,
    CONCAVE_TRANSFORMS,
    CONVEX_TRANSFORMS,
    LINEAR_TRANSFORMS,
    MONOTONE_TRANSFORMS,
    NONLOCAL_TRANSFORMS,
    NONMONOTONE_TRANSFORMS,
    NONSMOOTH_TRANSFORMS,
    POINTWISE_TRANSFORMS,
    SMOOTH_TRANSFORMS,
    TRANSFORM_REGISTRY,
    TRANSFORMS,
    TransformDefinition,
    TransformSpec,
    get_transform,
    get_transform_definition,
    get_transform_spec,
    list_transforms,
)

EXPECTED_REGISTRY_KEYS = tuple(sorted(TRANSFORMS))
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
    "climate",
    "ecological",
    "engineering",
    "environmental",
    "financial",
    "hydrology",
    "information",
    "mathematical",
    "medical",
    "spatial",
    "statistical",
    "temporal",
}

EXAMPLE_INPUTS: dict[str, np.ndarray] = {
    "scalar": np.array([-1.5, 0.2, 1.3, -0.7, 2.1], dtype=float),
    "functional": np.vstack(
        [
            np.sin(np.linspace(0.0, 2.0 * np.pi, 12)),
            np.cos(np.linspace(0.0, 2.0 * np.pi, 12)) + 0.1 * np.linspace(-1.0, 1.0, 12),
            np.linspace(-1.0, 1.0, 12) ** 3 - 0.2 * np.linspace(-1.0, 1.0, 12),
        ]
    ),
    "spatial": np.stack(
        [
            np.sin(
                np.pi
                * np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6), indexing="ij")[
                    0
                ]
            )
            * np.cos(
                np.pi
                * np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6), indexing="ij")[
                    1
                ]
            ),
            np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6), indexing="ij")[0] ** 2
            - np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6), indexing="ij")[1]
            + 0.25
            * np.sin(
                2.0
                * np.pi
                * np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6), indexing="ij")[
                    0
                ]
                * np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6), indexing="ij")[
                    1
                ]
            ),
            np.exp(
                -(
                    np.meshgrid(
                        np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6), indexing="ij"
                    )[0]
                    ** 2
                    + np.meshgrid(
                        np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6), indexing="ij"
                    )[1]
                    ** 2
                )
            )
            - 0.5
            * np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6), indexing="ij")[0],
        ],
        axis=0,
    ),
}


def _example_input(spec: TransformSpec) -> np.ndarray:
    for kind in ("spatial", "functional", "scalar"):
        if kind in spec.supported_output_kinds:
            return EXAMPLE_INPUTS[kind]
    raise AssertionError(f"No supported output kind for {spec.key}")


@pytest.mark.parametrize(
    ("key", "expected_module", "expected_mechanism"),
    [
        ("affine_a2_b1", "sabench.transforms.linear", "pointwise"),
        ("square_pointwise", "sabench.transforms.pointwise", "pointwise"),
        ("exp_pointwise", "sabench.transforms.pointwise", "pointwise"),
        ("relu_pointwise", "sabench.transforms.pointwise", "pointwise"),
        ("log1p_positive", "sabench.transforms.pointwise", "pointwise"),
        ("sqrt_abs", "sabench.transforms.pointwise", "pointwise"),
        ("abs_pointwise", "sabench.transforms.pointwise", "pointwise"),
        ("softplus_b01", "sabench.transforms.nonlinear", "pointwise"),
        ("temporal_cumsum", "sabench.transforms.samplewise", "samplewise"),
        ("temporal_peak", "sabench.transforms.aggregation", "aggregation"),
        ("gradient_magnitude", "sabench.transforms.field_ops", "field_op"),
        ("log_shift", "sabench.transforms.environmental", "samplewise"),
        ("regional_mean", "sabench.transforms.aggregation", "aggregation"),
        ("block_2x2", "sabench.transforms.aggregation", "aggregation"),
        ("exceedance_area", "sabench.transforms.aggregation", "aggregation"),
        ("matern_smooth", "sabench.transforms.field_ops", "field_op"),
        ("laplacian_roughness", "sabench.transforms.field_ops", "field_op"),
    ],
)
def test_transform_registry_keeps_expected_canonical_metadata(
    key: str,
    expected_module: str,
    expected_mechanism: str,
) -> None:
    spec = get_transform_spec(key)
    assert spec.module == expected_module
    assert spec.mechanism == expected_mechanism


def test_transform_registry_exposes_full_catalogue() -> None:
    assert list_transforms() == EXPECTED_REGISTRY_KEYS
    assert set(TRANSFORM_REGISTRY) == set(EXPECTED_REGISTRY_KEYS)
    assert len(TRANSFORM_REGISTRY) == len(EXPECTED_REGISTRY_KEYS) == 172


def test_transform_registry_entries_are_typed_and_callable() -> None:
    for key in EXPECTED_REGISTRY_KEYS:
        definition = get_transform_definition(key)
        assert isinstance(definition, TransformDefinition)
        assert isinstance(definition.spec, TransformSpec)
        assert callable(definition.transform)

        sample_input = _example_input(definition.spec)
        direct = definition.transform(sample_input)
        via_lookup = get_transform(key)(sample_input)
        assert isinstance(direct, np.ndarray)
        assert direct.shape[0] == sample_input.shape[0]
        np.testing.assert_allclose(direct, via_lookup)


def test_transform_spec_fields_are_valid() -> None:
    for key in EXPECTED_REGISTRY_KEYS:
        spec = get_transform_spec(key)
        assert spec.key == key
        assert spec.mechanism in VALID_MECHANISMS
        assert spec.supported_output_kinds
        assert set(spec.supported_output_kinds) <= VALID_OUTPUT_KINDS
        assert set(spec.tags) <= VALID_TAGS
        assert spec.function_name.startswith("t_")
        assert spec.reference


def test_transform_registry_can_filter_by_mechanism() -> None:
    pointwise = set(list_transforms(mechanism="pointwise"))
    samplewise = set(list_transforms(mechanism="samplewise"))
    aggregation = set(list_transforms(mechanism="aggregation"))
    field_op = set(list_transforms(mechanism="field_op"))

    assert {"affine_a2_b1", "softplus_b01", "tanh_a03"} <= pointwise
    assert "temporal_cumsum" in samplewise
    assert "temporal_peak" in aggregation
    assert "gradient_magnitude" in field_op

    union = pointwise | samplewise | aggregation | field_op
    assert union == set(EXPECTED_REGISTRY_KEYS)
    assert not (pointwise & samplewise)
    assert not (pointwise & aggregation)
    assert not (pointwise & field_op)
    assert not (samplewise & aggregation)
    assert not (samplewise & field_op)
    assert not (aggregation & field_op)


def test_unknown_transform_lookup_raises_clear_error() -> None:
    with pytest.raises(KeyError, match="Unknown transform"):
        get_transform_definition("does_not_exist")


def test_transform_registry_can_filter_by_tag() -> None:
    assert set(list_transforms(tag="pointwise")) == set(POINTWISE_TRANSFORMS)
    assert set(list_transforms(tag="linear")) == set(LINEAR_TRANSFORMS)
    assert set(list_transforms(tag="affine")) == set(AFFINE_TRANSFORMS)
    assert set(list_transforms(tag="nonlocal")) == set(NONLOCAL_TRANSFORMS)


def test_transform_registry_can_filter_by_output_kind() -> None:
    scalar = set(list_transforms(output_kind="scalar"))
    spatial = set(list_transforms(output_kind="spatial"))
    functional = set(list_transforms(output_kind="functional"))

    assert {"affine_a2_b1", "softplus_b01", "log_shift"} <= scalar
    assert {"affine_a2_b1", "gradient_magnitude", "regional_mean"} <= spatial
    assert {"temporal_cumsum", "temporal_peak", "log_shift"} <= functional
    assert "gradient_magnitude" not in scalar
    assert "gradient_magnitude" not in functional


def test_exported_property_sets_are_registry_derived() -> None:
    assert POINTWISE_TRANSFORMS == frozenset(list_transforms(tag="pointwise"))
    assert LINEAR_TRANSFORMS == frozenset(list_transforms(tag="linear"))
    assert AFFINE_TRANSFORMS == frozenset(list_transforms(tag="affine"))
    assert NONLOCAL_TRANSFORMS == frozenset(list_transforms(tag="nonlocal"))
    assert CONVEX_TRANSFORMS == frozenset(list_transforms(tag="convex"))
    assert CONCAVE_TRANSFORMS == frozenset(list_transforms(tag="concave"))
    assert MONOTONE_TRANSFORMS == frozenset(list_transforms(tag="monotone"))
    assert NONMONOTONE_TRANSFORMS == frozenset(list_transforms(tag="nonmonotone"))
    assert SMOOTH_TRANSFORMS == frozenset(list_transforms(tag="smooth"))
    assert NONSMOOTH_TRANSFORMS == frozenset(list_transforms(tag="nonsmooth"))
