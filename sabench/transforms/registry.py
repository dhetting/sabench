"""Canonical typed transform registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypedDict, cast

import numpy as np

from sabench.transforms.base import BoundTransform, TransformFunction
from sabench.transforms.catalog import (
    AFFINE_TRANSFORMS as CATALOG_AFFINE_TRANSFORMS,
)
from sabench.transforms.catalog import (
    CONCAVE_TRANSFORMS as CATALOG_CONCAVE_TRANSFORMS,
)
from sabench.transforms.catalog import (
    CONVEX_TRANSFORMS as CATALOG_CONVEX_TRANSFORMS,
)
from sabench.transforms.catalog import (
    LINEAR_TRANSFORMS as CATALOG_LINEAR_TRANSFORMS,
)
from sabench.transforms.catalog import (
    MONOTONE_TRANSFORMS as CATALOG_MONOTONE_TRANSFORMS,
)
from sabench.transforms.catalog import (
    NONLOCAL_TRANSFORMS as CATALOG_NONLOCAL_TRANSFORMS,
)
from sabench.transforms.catalog import (
    NONMONOTONE_TRANSFORMS as CATALOG_NONMONOTONE_TRANSFORMS,
)
from sabench.transforms.catalog import (
    NONSMOOTH_TRANSFORMS as CATALOG_NONSMOOTH_TRANSFORMS,
)
from sabench.transforms.catalog import (
    POINTWISE_TRANSFORMS as CATALOG_POINTWISE_TRANSFORMS,
)
from sabench.transforms.catalog import (
    SMOOTH_TRANSFORMS as CATALOG_SMOOTH_TRANSFORMS,
)
from sabench.transforms.catalog import (
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


class TransformCatalogMeta(TypedDict):
    """Typed view of transform catalog metadata entries."""

    name: str
    fn: TransformFunction
    params: dict[str, object]
    category: str
    reference: str


def _build_spatial_example() -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, 6)
    x_coord, y_coord = np.meshgrid(grid, grid, indexing="ij")
    return np.stack(
        [
            np.sin(np.pi * x_coord) * np.cos(np.pi * y_coord),
            x_coord**2 - y_coord + 0.25 * np.sin(2.0 * np.pi * x_coord * y_coord),
            np.exp(-(x_coord**2 + y_coord**2)) - 0.5 * x_coord,
        ],
        axis=0,
    )


EXAMPLE_INPUTS: dict[TransformOutputKind, np.ndarray] = {
    "scalar": np.array([-1.5, 0.2, 1.3, -0.7, 2.1], dtype=float),
    "functional": np.vstack(
        [
            np.sin(np.linspace(0.0, 2.0 * np.pi, 12)),
            np.cos(np.linspace(0.0, 2.0 * np.pi, 12)) + 0.1 * np.linspace(-1.0, 1.0, 12),
            np.linspace(-1.0, 1.0, 12) ** 3 - 0.2 * np.linspace(-1.0, 1.0, 12),
        ]
    ),
    "spatial": _build_spatial_example(),
}

_DOMAIN_TAGS: dict[str, TransformTag] = {
    "climate": "climate",
    "ecological": "ecological",
    "engineering": "engineering",
    "environmental": "environmental",
    "financial": "financial",
    "hydrology": "hydrology",
    "information": "information",
    "mathematical": "mathematical",
    "medical": "medical",
    "spatial": "spatial",
    "statistical": "statistical",
    "temporal": "temporal",
}

_PROPERTY_TAG_SOURCES: tuple[tuple[TransformTag, set[str]], ...] = (
    ("affine", CATALOG_AFFINE_TRANSFORMS),
    ("linear", CATALOG_LINEAR_TRANSFORMS),
    ("pointwise", CATALOG_POINTWISE_TRANSFORMS),
    ("nonlocal", CATALOG_NONLOCAL_TRANSFORMS),
    ("convex", CATALOG_CONVEX_TRANSFORMS),
    ("concave", CATALOG_CONCAVE_TRANSFORMS),
    ("monotone", CATALOG_MONOTONE_TRANSFORMS),
    ("nonmonotone", CATALOG_NONMONOTONE_TRANSFORMS),
    ("smooth", CATALOG_SMOOTH_TRANSFORMS),
    ("nonsmooth", CATALOG_NONSMOOTH_TRANSFORMS),
)

_FIELD_OP_KEYS: frozenset[str] = frozenset(
    {
        "contour_exceedance",
        "gradient_magnitude",
        "laplacian_roughness",
        "matern_smooth",
    }
)


def _get_catalog_meta(key: str) -> TransformCatalogMeta:
    """Return a typed view of a transform catalog metadata entry."""
    return cast(TransformCatalogMeta, TRANSFORMS[key])


def _bind_transform(meta: TransformCatalogMeta) -> BoundTransform:
    """Bind default parameters to a registered transform implementation."""
    fn = meta["fn"]
    params = meta["params"]

    def bound_transform(y: np.ndarray) -> np.ndarray:
        return fn(y, **params)

    return bound_transform


def _broadcasts_sample_constant(output: np.ndarray) -> bool:
    """Return whether each sample is constant across its output support."""
    flat = output.reshape(len(output), -1)
    return np.allclose(flat, flat[:, :1])


def _infer_supported_output_kinds(key: str, category: str) -> tuple[TransformOutputKind, ...]:
    """Infer supported output kinds from canonical catalog categories."""
    if category == "spatial":
        return ("spatial",)
    if category == "temporal" or key.startswith("temporal_"):
        return ("functional",)
    return ("scalar", "spatial", "functional")


def _infer_mechanism(
    key: str,
    category: str,
    transform: BoundTransform,
    supported_output_kinds: tuple[TransformOutputKind, ...],
) -> TransformMechanism:
    """Infer the typed transform mechanism from registered transform behavior."""
    if key in CATALOG_POINTWISE_TRANSFORMS:
        return "pointwise"

    if category == "spatial":
        return "field_op" if key in _FIELD_OP_KEYS else "aggregation"

    exemplar_kind: TransformOutputKind = "functional"
    if exemplar_kind not in supported_output_kinds:
        exemplar_kind = supported_output_kinds[0]

    exemplar = EXAMPLE_INPUTS[exemplar_kind]
    output = transform(exemplar)
    if output.shape != exemplar.shape or _broadcasts_sample_constant(output):
        return "aggregation"
    return "samplewise"


def _collect_tags(key: str, category: str) -> tuple[TransformTag, ...]:
    """Derive canonical transform tags from the canonical catalog metadata."""
    tags: list[TransformTag] = []
    for tag_name, keys in _PROPERTY_TAG_SOURCES:
        if key in keys:
            tags.append(tag_name)
    if key not in CATALOG_LINEAR_TRANSFORMS:
        tags.append("nonlinear")
    tags.append(_DOMAIN_TAGS[category])
    return tuple(tags)


def _build_spec(key: str) -> TransformSpec:
    """Build canonical typed metadata from the transform catalog."""
    meta = _get_catalog_meta(key)
    transform = _bind_transform(meta)
    category = meta["category"]
    fn = meta["fn"]
    supported_output_kinds = _infer_supported_output_kinds(key, category)

    return TransformSpec(
        key=key,
        name=meta["name"],
        mechanism=_infer_mechanism(key, category, transform, supported_output_kinds),
        module=fn.__module__,
        function_name=fn.__name__,
        supported_output_kinds=supported_output_kinds,
        tags=_collect_tags(key, category),
        reference=meta["reference"],
    )


_REGISTRY: dict[str, TransformDefinition] = {
    key: TransformDefinition(
        spec=_build_spec(key), transform=_bind_transform(_get_catalog_meta(key))
    )
    for key in TRANSFORMS
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


def list_transforms(
    mechanism: TransformMechanism | None = None,
    tag: TransformTag | None = None,
    output_kind: TransformOutputKind | None = None,
) -> tuple[str, ...]:
    """List registered transform keys, optionally filtered by typed metadata."""
    keys = [
        key
        for key, definition in TRANSFORM_REGISTRY.items()
        if (mechanism is None or definition.spec.mechanism == mechanism)
        and (tag is None or tag in definition.spec.tags)
        and (output_kind is None or output_kind in definition.spec.supported_output_kinds)
    ]
    return tuple(sorted(keys))


AFFINE_TRANSFORMS: frozenset[str] = frozenset(list_transforms(tag="affine"))
LINEAR_TRANSFORMS: frozenset[str] = frozenset(list_transforms(tag="linear"))
POINTWISE_TRANSFORMS: frozenset[str] = frozenset(list_transforms(tag="pointwise"))
NONLOCAL_TRANSFORMS: frozenset[str] = frozenset(list_transforms(tag="nonlocal"))
CONVEX_TRANSFORMS: frozenset[str] = frozenset(list_transforms(tag="convex"))
CONCAVE_TRANSFORMS: frozenset[str] = frozenset(list_transforms(tag="concave"))
MONOTONE_TRANSFORMS: frozenset[str] = frozenset(list_transforms(tag="monotone"))
NONMONOTONE_TRANSFORMS: frozenset[str] = frozenset(list_transforms(tag="nonmonotone"))
SMOOTH_TRANSFORMS: frozenset[str] = frozenset(list_transforms(tag="smooth"))
NONSMOOTH_TRANSFORMS: frozenset[str] = frozenset(list_transforms(tag="nonsmooth"))
