"""
sabench.transforms — Registry and utilities for output transformations.

This module provides physically-motivated nonlinear and linear output
transformations for investigating Sobol index non-commutativity across
environmental, engineering, spatial, temporal, and statistical domains.
"""

from sabench.transforms.registry import (
    TRANSFORM_SPECS,
    get_transform,
    get_transform_spec,
    list_transform_names,
)
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
    apply_transform,
    score_transform,
)
from sabench.transforms.types import TransformCategory, TransformSpec, TransformTag

__all__ = [
    "TRANSFORM_SPECS",
    "TransformCategory",
    "TransformSpec",
    "TransformTag",
    "get_transform",
    "get_transform_spec",
    "list_transform_names",
    "TRANSFORMS",
    "LINEAR_TRANSFORMS",
    "POINTWISE_TRANSFORMS",
    "AFFINE_TRANSFORMS",
    "NONLOCAL_TRANSFORMS",
    "CONVEX_TRANSFORMS",
    "CONCAVE_TRANSFORMS",
    "MONOTONE_TRANSFORMS",
    "NONMONOTONE_TRANSFORMS",
    "SMOOTH_TRANSFORMS",
    "NONSMOOTH_TRANSFORMS",
    "apply_transform",
    "score_transform",
]
