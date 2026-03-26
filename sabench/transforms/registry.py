"""Canonical typed registry for sabench transforms."""

from __future__ import annotations

from sabench.benchmarks.types import OutputKind
from sabench.transforms.base import TransformFunction
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
from sabench.transforms.types import TransformCategory, TransformSpec, TransformTag

_ALL_OUTPUT_KINDS: frozenset[OutputKind] = frozenset(OutputKind)
_TAG_MEMBERSHIP: tuple[tuple[TransformTag, set[str]], ...] = (
    (TransformTag.AFFINE, AFFINE_TRANSFORMS),
    (TransformTag.CONCAVE, CONCAVE_TRANSFORMS),
    (TransformTag.CONVEX, CONVEX_TRANSFORMS),
    (TransformTag.LINEAR, LINEAR_TRANSFORMS),
    (TransformTag.MONOTONE, MONOTONE_TRANSFORMS),
    (TransformTag.NONLOCAL, NONLOCAL_TRANSFORMS),
    (TransformTag.NONMONOTONE, NONMONOTONE_TRANSFORMS),
    (TransformTag.NONSMOOTH, NONSMOOTH_TRANSFORMS),
    (TransformTag.POINTWISE, POINTWISE_TRANSFORMS),
    (TransformTag.SMOOTH, SMOOTH_TRANSFORMS),
)


def _category_from_legacy_label(label: str) -> TransformCategory:
    try:
        return TransformCategory(label)
    except ValueError as exc:
        msg = f"Unsupported transform category label: {label!r}"
        raise ValueError(msg) from exc


def _tags_for(key: str) -> frozenset[TransformTag]:
    tags = {tag for tag, members in _TAG_MEMBERSHIP if key in members}
    if TransformTag.AFFINE in tags:
        tags.add(TransformTag.LINEAR)
    return frozenset(tags)


def _supported_output_kinds_for(key: str) -> frozenset[OutputKind]:
    """Return conservative runtime-compatible output kinds for Slice 6.

    The legacy monolith largely implements transforms in array-shape-generic
    form, so the first typed registry records broad runtime compatibility.
    Later transform-family slices can tighten this contract where needed.
    """

    del key
    return _ALL_OUTPUT_KINDS


def _spec_from_legacy_entry(key: str, payload: dict[str, object]) -> TransformSpec:
    transform_fn = payload["fn"]
    if not isinstance(transform_fn, TransformFunction):
        msg = f"Registered transform {key!r} does not expose a callable transform function."
        raise TypeError(msg)

    params_default = payload.get("params", {})
    if not isinstance(params_default, dict):
        msg = f"Registered transform {key!r} exposes non-dict params payload."
        raise TypeError(msg)

    return TransformSpec(
        key=key,
        name=str(payload["name"]),
        transform_fn=transform_fn,
        category=_category_from_legacy_label(str(payload["category"])),
        tags=_tags_for(key),
        supported_output_kinds=_supported_output_kinds_for(key),
        params_default=dict(params_default),
        module=transform_fn.__module__,
    )


_TRANSFORM_SPECS = tuple(
    _spec_from_legacy_entry(key, payload) for key, payload in TRANSFORMS.items()
)

TRANSFORM_SPECS: dict[str, TransformSpec] = {spec.key: spec for spec in _TRANSFORM_SPECS}
if len(TRANSFORM_SPECS) != len(_TRANSFORM_SPECS):
    raise ValueError("Transform registry contains duplicate keys.")


def get_transform_spec(key: str) -> TransformSpec:
    """Return the canonical typed spec for a transform key."""

    return TRANSFORM_SPECS[key]


def get_transform(key: str) -> TransformFunction:
    """Return the callable transform registered under ``key``."""

    return get_transform_spec(key).transform_fn


def list_transform_names() -> list[str]:
    """Return transform keys in deterministic sorted order."""

    return sorted(TRANSFORM_SPECS)
