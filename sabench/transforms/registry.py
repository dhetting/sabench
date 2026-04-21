"""Canonical typed transform registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypedDict, cast

import numpy as np

from sabench.transforms.base import BoundTransform, TransformFunction
from sabench.transforms.transforms import (
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
    TRANSFORMS,
)
from sabench.transforms.types import (
    TransformMechanism,
    TransformOutputKind,
    TransformSpec,
    TransformTag,
)


@dataclass(frozen=True, slots=True)
class TransformDefinition:
    """Canonical registry entry for a transform."""

    spec: TransformSpec
    transform: BoundTransform


class LegacyTransformMeta(TypedDict):
    """Typed view of legacy transform metadata entries."""

    name: str
    fn: TransformFunction
    params: dict[str, object]
    category: str
    reference: str


def _get_legacy_meta(key: str) -> LegacyTransformMeta:
    """Return a typed view of a legacy transform metadata entry."""
    return cast(LegacyTransformMeta, TRANSFORMS[key])


_MECHANISMS: dict[str, TransformMechanism] = {
    "affine_a2_b1": "pointwise",
    "tanh_a03": "pointwise",
    "softplus_b01": "pointwise",
    "temporal_cumsum": "samplewise",
    "temporal_peak": "aggregation",
    "gradient_magnitude": "field_op",
}

_SUPPORTED_OUTPUT_KINDS: dict[str, tuple[TransformOutputKind, ...]] = {
    "affine_a2_b1": ("scalar", "spatial", "functional"),
    "tanh_a03": ("scalar", "spatial", "functional"),
    "softplus_b01": ("scalar", "spatial", "functional"),
    "temporal_cumsum": ("functional",),
    "temporal_peak": ("functional",),
    "gradient_magnitude": ("spatial",),
}

_PROPERTY_TAG_SOURCES: tuple[tuple[TransformTag, set[str]], ...] = (
    ("affine", AFFINE_TRANSFORMS),
    ("linear", LINEAR_TRANSFORMS),
    ("pointwise", POINTWISE_TRANSFORMS),
    ("nonlocal", NONLOCAL_TRANSFORMS),
    ("convex", CONVEX_TRANSFORMS),
    ("concave", CONCAVE_TRANSFORMS),
    ("monotone", MONOTONE_TRANSFORMS),
    ("nonmonotone", NONMONOTONE_TRANSFORMS),
    ("smooth", SMOOTH_TRANSFORMS),
    ("nonsmooth", NONSMOOTH_TRANSFORMS),
)

_DOMAIN_TAGS: dict[str, TransformTag] = {
    "environmental": "environmental",
    "engineering": "engineering",
    "spatial": "spatial",
    "temporal": "temporal",
    "statistical": "statistical",
    "mathematical": "mathematical",
}


def _bind_transform(meta: LegacyTransformMeta) -> BoundTransform:
    """Bind default parameters to a registered transform implementation."""
    fn = meta["fn"]
    params = meta["params"]

    def bound_transform(y: np.ndarray) -> np.ndarray:
        return fn(y, **params)

    return bound_transform


def _collect_tags(key: str, category: str) -> tuple[TransformTag, ...]:
    """Derive canonical transform tags from the legacy registry metadata."""
    tags: list[TransformTag] = []
    for tag_name, keys in _PROPERTY_TAG_SOURCES:
        if key in keys:
            tags.append(tag_name)
    if key not in LINEAR_TRANSFORMS:
        tags.append("nonlinear")
    tags.append(_DOMAIN_TAGS[category])
    return tuple(tags)


def _build_spec(key: str) -> TransformSpec:
    """Build canonical typed metadata from the legacy transform registry."""
    meta = _get_legacy_meta(key)
    fn = meta["fn"]
    category = meta["category"]
    name = meta["name"]
    reference = meta["reference"]

    return TransformSpec(
        key=key,
        name=name,
        mechanism=_MECHANISMS[key],
        module=fn.__module__,
        function_name=fn.__name__,
        supported_output_kinds=_SUPPORTED_OUTPUT_KINDS[key],
        tags=_collect_tags(key, category),
        reference=reference,
    )


_REPRESENTATIVE_KEYS: tuple[str, ...] = (
    "affine_a2_b1",
    "tanh_a03",
    "softplus_b01",
    "temporal_cumsum",
    "temporal_peak",
    "gradient_magnitude",
)

_REGISTRY: dict[str, TransformDefinition] = {
    key: TransformDefinition(
        spec=_build_spec(key), transform=_bind_transform(_get_legacy_meta(key))
    )
    for key in _REPRESENTATIVE_KEYS
}

TRANSFORM_REGISTRY: Mapping[str, TransformDefinition] = MappingProxyType(_REGISTRY)


def get_transform_definition(key: str) -> TransformDefinition:
    """Return the registry entry for a transform key."""
    try:
        return TRANSFORM_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown transform: {key}") from exc


def get_transform_spec(key: str) -> TransformSpec:
    """Return the typed spec for a transform key."""
    return get_transform_definition(key).spec


def get_transform(key: str) -> BoundTransform:
    """Return a callable transform with default parameters bound."""
    return get_transform_definition(key).transform


def list_transforms(mechanism: TransformMechanism | None = None) -> tuple[str, ...]:
    """List registered transform keys, optionally filtered by mechanism."""
    keys = [
        key
        for key, definition in TRANSFORM_REGISTRY.items()
        if mechanism is None or definition.spec.mechanism == mechanism
    ]
    return tuple(sorted(keys))
