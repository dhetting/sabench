"""Typed metadata definitions for sabench transforms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from sabench.benchmarks.types import OutputKind
from sabench.transforms.base import TransformFunction


class TransformCategory(str, Enum):
    """Scientific/domain categories used by the legacy transform catalog."""

    CLIMATE = "climate"
    ECOLOGICAL = "ecological"
    ENGINEERING = "engineering"
    ENVIRONMENTAL = "environmental"
    FINANCIAL = "financial"
    HYDROLOGY = "hydrology"
    INFORMATION = "information"
    MATHEMATICAL = "mathematical"
    MEDICAL = "medical"
    SPATIAL = "spatial"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"


class TransformTag(str, Enum):
    """Canonical typed property tags for the transform registry."""

    AFFINE = "affine"
    CONCAVE = "concave"
    CONVEX = "convex"
    LINEAR = "linear"
    MONOTONE = "monotone"
    NONLOCAL = "nonlocal"
    NONMONOTONE = "nonmonotone"
    NONSMOOTH = "nonsmooth"
    POINTWISE = "pointwise"
    SMOOTH = "smooth"


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Canonical typed definition for a transform registry entry."""

    key: str
    name: str
    transform_fn: TransformFunction
    category: TransformCategory
    tags: frozenset[TransformTag]
    supported_output_kinds: frozenset[OutputKind]
    params_default: Mapping[str, object]
    module: str
