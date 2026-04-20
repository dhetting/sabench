"""Typed transform metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TransformMechanism = Literal["pointwise", "samplewise", "aggregation", "field_op"]
TransformTag = Literal[
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
]
TransformOutputKind = Literal["scalar", "spatial", "functional"]


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Canonical typed metadata for a transform definition."""

    key: str
    name: str
    mechanism: TransformMechanism
    module: str
    function_name: str
    supported_output_kinds: tuple[TransformOutputKind, ...]
    tags: tuple[TransformTag, ...]
    reference: str
