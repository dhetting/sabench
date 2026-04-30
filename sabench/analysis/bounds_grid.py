"""Registry-driven bounds-theorem grid execution utilities.

This module provides the reusable execution layer for the bounds-theorem
notebook. It classifies benchmark/transform pairs by theorem assumptions,
computes Taylor-reference diagnostics for supported scalar pointwise pairs, and
labels sample-range calculations as diagnostics unless caller-supplied support
bounds are provided.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from sabench.analysis.bounds import (
    BoundsApplicabilityStatus,
    classify_bounds_applicability,
    get_smooth_pointwise_analysis,
    local_affine_diagnostics,
    projection_perturbation_bound,
    taylor_reference_diagnostics,
)
from sabench.analysis.estimators import jansen_s1_st
from sabench.analysis.grid import variance_weighted_sobol_profile
from sabench.benchmarks.registry import BENCHMARK_REGISTRY, get_benchmark_definition
from sabench.benchmarks.types import BenchmarkOutputKind
from sabench.transforms.registry import TRANSFORM_REGISTRY, get_transform_definition
from sabench.transforms.types import TransformMechanism, TransformTag

BoundsStatus = Literal[
    "bounds_supported",
    "bounds_diagnostic_sample_support",
    "bounds_not_scalar_output",
    "bounds_not_pointwise",
    "bounds_not_smooth",
    "bounds_no_derivative_metadata",
    "bounds_domain_invalid",
    "bounds_reference_zero_variance",
    "bounds_eta_ge_one",
    "bounds_failed",
]

BOUNDS_STATUSES: tuple[BoundsStatus, ...] = (
    "bounds_supported",
    "bounds_diagnostic_sample_support",
    "bounds_not_scalar_output",
    "bounds_not_pointwise",
    "bounds_not_smooth",
    "bounds_no_derivative_metadata",
    "bounds_domain_invalid",
    "bounds_reference_zero_variance",
    "bounds_eta_ge_one",
    "bounds_failed",
)


@dataclass(frozen=True, slots=True)
class BoundsPairClassification:
    """Static bounds-theorem classification for one benchmark/transform pair."""

    benchmark_key: str
    transform_key: str
    benchmark_output_kind: BenchmarkOutputKind
    transform_mechanism: TransformMechanism
    transform_tags: tuple[TransformTag, ...]
    static_status: BoundsApplicabilityStatus
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return a tabular representation of the static classification."""
        values = asdict(self)
        values["transform_tags"] = ";".join(self.transform_tags)
        return values


@dataclass(frozen=True, slots=True)
class BoundsGridResult:
    """Result row for one bounds-theorem benchmark/transform evaluation."""

    benchmark_key: str
    transform_key: str
    bounds_status: BoundsStatus
    reason: str
    benchmark_output_kind: BenchmarkOutputKind
    transform_mechanism: TransformMechanism
    transform_tags: tuple[TransformTag, ...]
    n_base: int
    seed: int | None
    n_inputs: int
    n_evaluations: int | None = None
    output_shape: tuple[int, ...] | None = None
    output_finite: bool | None = None
    output_variance: float | None = None
    taylor_order: int | None = None
    diagnostics: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a flat dictionary suitable for notebook tables."""
        values = asdict(self)
        diagnostics = values.pop("diagnostics") or {}
        values["transform_tags"] = ";".join(self.transform_tags)
        values["output_shape"] = _shape_to_string(self.output_shape)
        values.update(diagnostics)
        return values


def classify_bounds_grid_pair(
    benchmark_key: str,
    transform_key: str,
) -> BoundsPairClassification:
    """Classify whether a pair satisfies static bounds-theorem assumptions."""
    benchmark_definition = get_benchmark_definition(benchmark_key)
    transform_definition = get_transform_definition(transform_key)
    benchmark_output_kind = benchmark_definition.spec.output_kind
    transform_spec = transform_definition.spec
    applicability = classify_bounds_applicability(
        output_kind=benchmark_output_kind,
        mechanism=transform_spec.mechanism,
        tags=transform_spec.tags,
        transform_key=transform_key,
    )
    return BoundsPairClassification(
        benchmark_key=benchmark_key,
        transform_key=transform_key,
        benchmark_output_kind=benchmark_output_kind,
        transform_mechanism=transform_spec.mechanism,
        transform_tags=transform_spec.tags,
        static_status=applicability.status,
        reason=applicability.reason,
    )


def evaluate_bounds_pair(
    benchmark_key: str,
    transform_key: str,
    *,
    n_base: int = 512,
    seed: int | None = 0,
    taylor_order: int = 2,
    support: tuple[float, float] | None = None,
    clip_sobol: bool = True,
) -> BoundsGridResult:
    """Evaluate one benchmark/transform pair for bounds-theorem diagnostics."""
    _validate_n_base(n_base)
    classification = classify_bounds_grid_pair(benchmark_key, transform_key)
    benchmark_definition = get_benchmark_definition(benchmark_key)
    benchmark = benchmark_definition.benchmark_cls()

    if classification.static_status != "bounds_supported":
        return _result_from_classification(
            classification,
            bounds_status=classification.static_status,
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
        )

    sample = benchmark.sample(n_base, seed=seed)
    try:
        output = np.asarray(benchmark.evaluate(sample), dtype=float)
    except Exception as exc:  # pragma: no cover - defensive row-level failure path
        return _result_from_classification(
            classification,
            bounds_status="bounds_failed",
            reason=str(exc),
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
            taylor_order=taylor_order,
        )

    output_error = _validate_scalar_output(output, expected_rows=sample.shape[0])
    if output_error is not None:
        return _result_from_classification(
            classification,
            bounds_status="bounds_failed",
            reason=output_error,
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
            n_evaluations=sample.shape[0],
            output_shape=output.shape,
            output_finite=bool(np.all(np.isfinite(output))),
            taylor_order=taylor_order,
        )

    analysis = get_smooth_pointwise_analysis(transform_key)
    try:
        taylor = taylor_reference_diagnostics(
            output,
            analysis,
            order=taylor_order,
            support=support,
        )
        local_affine = local_affine_diagnostics(output, analysis, support=support)
    except ValueError as exc:
        return _result_from_classification(
            classification,
            bounds_status="bounds_domain_invalid",
            reason=str(exc),
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
            n_evaluations=sample.shape[0],
            output_shape=output.shape,
            output_finite=True,
            output_variance=_empirical_variance(output),
            taylor_order=taylor_order,
        )
    except Exception as exc:  # pragma: no cover - defensive row-level failure path
        return _result_from_classification(
            classification,
            bounds_status="bounds_failed",
            reason=str(exc),
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
            n_evaluations=sample.shape[0],
            output_shape=output.shape,
            output_finite=True,
            output_variance=_empirical_variance(output),
            taylor_order=taylor_order,
        )

    bounds_status = _status_from_taylor(taylor.status, taylor.support_source)
    diagnostics = {
        **taylor.as_summary_dict(),
        **local_affine.as_summary_dict(),
    }
    diagnostics["taylor_status"] = taylor.status
    if taylor.eta_empirical is not None and taylor.eta_empirical < 1.0:
        diagnostics.update(
            _projection_bound_summaries(
                taylor.reference_values,
                analysis.transform(output),
                benchmark.d,
                n_base,
                taylor.eta_empirical,
                clip_sobol=clip_sobol,
            )
        )

    return _result_from_classification(
        classification,
        bounds_status=bounds_status,
        reason=_reason_from_status(bounds_status),
        n_base=n_base,
        seed=seed,
        n_inputs=benchmark.d,
        n_evaluations=sample.shape[0],
        output_shape=output.shape,
        output_finite=True,
        output_variance=_empirical_variance(output),
        taylor_order=taylor_order,
        diagnostics=diagnostics,
    )


def evaluate_bounds_grid(
    benchmark_keys: Iterable[str] | None = None,
    transform_keys: Iterable[str] | None = None,
    *,
    n_base: int = 512,
    seed: int | None = 0,
    taylor_order: int = 2,
    support_by_pair: Mapping[tuple[str, str], tuple[float, float]] | None = None,
    clip_sobol: bool = True,
) -> tuple[BoundsGridResult, ...]:
    """Evaluate a deterministic bounds-theorem benchmark/transform grid."""
    benchmarks = tuple(benchmark_keys) if benchmark_keys is not None else tuple(BENCHMARK_REGISTRY)
    transforms = tuple(transform_keys) if transform_keys is not None else tuple(TRANSFORM_REGISTRY)
    supports = support_by_pair or {}
    return tuple(
        evaluate_bounds_pair(
            benchmark_key,
            transform_key,
            n_base=n_base,
            seed=seed,
            taylor_order=taylor_order,
            support=supports.get((benchmark_key, transform_key)),
            clip_sobol=clip_sobol,
        )
        for benchmark_key in benchmarks
        for transform_key in transforms
    )


def _projection_bound_summaries(
    reference_values: np.ndarray,
    transformed_values: np.ndarray,
    n_inputs: int,
    n_base: int,
    eta: float,
    *,
    clip_sobol: bool,
) -> dict[str, Any]:
    reference_s1, reference_st = jansen_s1_st(
        reference_values,
        N=n_base,
        d=n_inputs,
        clip=clip_sobol,
    )
    transformed_s1, transformed_st = jansen_s1_st(
        transformed_values,
        N=n_base,
        d=n_inputs,
        clip=clip_sobol,
    )
    reference_s1_profile = variance_weighted_sobol_profile(reference_s1, reference_values)
    reference_st_profile = variance_weighted_sobol_profile(reference_st, reference_values)
    transformed_s1_profile = variance_weighted_sobol_profile(transformed_s1, transformed_values)
    transformed_st_profile = variance_weighted_sobol_profile(transformed_st, transformed_values)
    bound_s1 = projection_perturbation_bound(eta, reference_s1_profile, cap=True)
    bound_st = projection_perturbation_bound(eta, reference_st_profile, cap=True)
    shift_s1 = np.abs(transformed_s1_profile - reference_s1_profile)
    shift_st = np.abs(transformed_st_profile - reference_st_profile)
    return {
        "projection_bound_s1_max": float(np.max(bound_s1)),
        "projection_bound_st_max": float(np.max(bound_st)),
        "projection_bound_s1_mean": float(np.mean(bound_s1)),
        "projection_bound_st_mean": float(np.mean(bound_st)),
        "reference_shift_s1_max": float(np.max(shift_s1)),
        "reference_shift_st_max": float(np.max(shift_st)),
        "reference_shift_s1_mean": float(np.mean(shift_s1)),
        "reference_shift_st_mean": float(np.mean(shift_st)),
    }


def _result_from_classification(
    classification: BoundsPairClassification,
    *,
    bounds_status: BoundsStatus,
    n_base: int,
    seed: int | None,
    n_inputs: int,
    reason: str | None = None,
    n_evaluations: int | None = None,
    output_shape: tuple[int, ...] | None = None,
    output_finite: bool | None = None,
    output_variance: float | None = None,
    taylor_order: int | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> BoundsGridResult:
    return BoundsGridResult(
        benchmark_key=classification.benchmark_key,
        transform_key=classification.transform_key,
        bounds_status=bounds_status,
        reason=classification.reason if reason is None else reason,
        benchmark_output_kind=classification.benchmark_output_kind,
        transform_mechanism=classification.transform_mechanism,
        transform_tags=classification.transform_tags,
        n_base=n_base,
        seed=seed,
        n_inputs=n_inputs,
        n_evaluations=n_evaluations,
        output_shape=output_shape,
        output_finite=output_finite,
        output_variance=output_variance,
        taylor_order=taylor_order,
        diagnostics=diagnostics,
    )


def _status_from_taylor(taylor_status: str, support_source: str) -> BoundsStatus:
    if taylor_status == "reference_zero_variance":
        return "bounds_reference_zero_variance"
    if taylor_status == "eta_ge_one":
        return "bounds_eta_ge_one"
    if support_source == "provided_support":
        return "bounds_supported"
    return "bounds_diagnostic_sample_support"


def _reason_from_status(status: BoundsStatus) -> str:
    if status == "bounds_supported":
        return "Taylor-reference diagnostics computed with provided support bounds."
    if status == "bounds_diagnostic_sample_support":
        return "Taylor-reference diagnostics computed with empirical sample-range support."
    if status == "bounds_reference_zero_variance":
        return "Taylor reference has zero empirical variance."
    if status == "bounds_eta_ge_one":
        return "Taylor residual ratio eta is greater than or equal to one."
    return "Bounds diagnostics failed."


def _validate_n_base(n_base: int) -> None:
    if n_base <= 0:
        raise ValueError("n_base must be positive")


def _validate_scalar_output(output: np.ndarray, *, expected_rows: int) -> str | None:
    if output.ndim != 1:
        return "bounds diagnostics require one-dimensional scalar benchmark output"
    if output.shape[0] != expected_rows:
        return f"output sample axis has length {output.shape[0]}, expected {expected_rows}"
    if not np.all(np.isfinite(output)):
        return "output contains nonfinite values"
    return None


def _empirical_variance(values: np.ndarray) -> float:
    return float(np.var(np.asarray(values, dtype=float)))


def _shape_to_string(shape: tuple[int, ...] | None) -> str | None:
    if shape is None:
        return None
    return "x".join(str(part) for part in shape)
