"""Registry-driven bounds-theorem grid execution utilities.

This module provides the reusable execution layer for the bounds-theorem
notebook. It classifies benchmark/transform pairs by theorem assumptions,
computes Taylor-reference diagnostics for supported scalar pointwise pairs, and
labels sample-range calculations as diagnostics unless caller-supplied support
bounds are provided.

``BENCHMARK_OUTPUT_BOUNDS`` contains analytically derived output bounds for a
subset of scalar benchmarks. These bounds are guaranteed to contain all possible
output values for the default benchmark parameterisation, allowing them to be
used as theorem-backed support in ``evaluate_bounds_grid``.
"""

from __future__ import annotations

import math
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

# ---------------------------------------------------------------------------
# Analytically-derived output bounds for scalar benchmarks
# ---------------------------------------------------------------------------
# Each entry gives the closed-form minimum and maximum of the benchmark output
# over the full declared input domain under the default parameterisation.
# These bounds are theorem-backed: they contain every possible output value,
# so passing them to ``evaluate_bounds_grid`` as ``benchmark_support`` promotes
# qualifying pairs to ``bounds_supported`` status.
#
# Derivations (default parameterisations)
# ----------------------------------------
# Ishigami (a=7, b=0.1, X∈[-π,π]^3):
#   Y = sin(X1) + a·sin²(X2) + b·X3⁴·sin(X1)
#   Y_min = -(1 + b·π⁴),  Y_max = 1 + a + b·π⁴
#
# SobolG (a=[0,1,4.5,9,99,99,99,99], X∈[0,1]^8):
#   g_i = (|4x-2| + a_i)/(1+a_i);  G = ∏ g_i
#   G_min = ∏ a_i/(1+a_i) = 0,  G_max = ∏ (2+a_i)/(1+a_i)
#
# LinearModel (a=[3,2,1,0.5,0.1], X∈[0,1]^5):
#   Y = X·a;  Y_min = 0,  Y_max = Σa = 6.6
#
# AdditiveQuadratic (d=5, a=linspace(2,0.2,5), b=ones(5), X∈[0,1]^5):
#   Y = Σ(a_i·X_i² + b_i·X_i);  Y_min = 0,  Y_max = Σ(a_i+b_i)
#
# CornerPeak (d=6, c_i = i/(6·7/2), X∈[0,1]^6):
#   Y = (1 + Σ c_i·X_i)^{-(d+1)};
#   Y_min = (1 + Σ c_i)^{-7},  Y_max = 1

_ISHIGAMI_B = 0.1
_ISHIGAMI_A = 7.0
_SOBOLG_A = np.array([0.0, 1.0, 4.5, 9.0, 99.0, 99.0, 99.0, 99.0])
_CORNERPEAK_D = 6
_CORNERPEAK_C = np.arange(1, _CORNERPEAK_D + 1, dtype=float) / (
    _CORNERPEAK_D * (_CORNERPEAK_D + 1) / 2
)
_ADDQUAD_D = 5
_ADDQUAD_A = np.linspace(2.0, 0.2, _ADDQUAD_D)
_ADDQUAD_B = np.ones(_ADDQUAD_D)

BENCHMARK_OUTPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "Ishigami": (
        -(1.0 + _ISHIGAMI_B * math.pi**4),
        1.0 + _ISHIGAMI_A + _ISHIGAMI_B * math.pi**4,
    ),
    "SobolG": (
        float(np.prod(_SOBOLG_A / (1.0 + _SOBOLG_A))),
        float(np.prod((2.0 + _SOBOLG_A) / (1.0 + _SOBOLG_A))),
    ),
    "LinearModel": (0.0, float(np.array([3.0, 2.0, 1.0, 0.5, 0.1]).sum())),
    "AdditiveQuadratic": (0.0, float((_ADDQUAD_A + _ADDQUAD_B).sum())),
    "CornerPeak": (
        float((1.0 + float(_CORNERPEAK_C.sum())) ** (-_CORNERPEAK_D - 1)),
        1.0,
    ),
}
"""Analytically-derived output bounds for scalar benchmarks.

Maps benchmark key → ``(y_lower, y_upper)`` where the interval contains every
possible output value for the default parameterisation.  Pass this mapping to
``evaluate_bounds_grid`` via the ``benchmark_support`` argument to promote
qualifying pairs from ``bounds_diagnostic_sample_support`` to
``bounds_supported``.
"""

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
    benchmark_support: Mapping[str, tuple[float, float]] | None = None,
    clip_sobol: bool = True,
) -> tuple[BoundsGridResult, ...]:
    """Evaluate a deterministic bounds-theorem benchmark/transform grid.

    Parameters
    ----------
    benchmark_keys:
        Benchmark keys to include; defaults to all registered benchmarks.
    transform_keys:
        Transform keys to include; defaults to all registered transforms.
    n_base:
        Sample size per Saltelli block.
    seed:
        Random seed for reproducibility.
    taylor_order:
        Taylor expansion order for the reference computation.
    support_by_pair:
        Per-pair theorem-backed support bounds ``(y_lower, y_upper)``.
        When supplied for a pair the status is promoted to
        ``bounds_supported``; takes precedence over ``benchmark_support``.
    benchmark_support:
        Per-benchmark theorem-backed support bounds ``(y_lower, y_upper)``.
        Applied to all transform keys for each listed benchmark unless an
        explicit entry is already present in ``support_by_pair``.  Pairs
        promoted this way receive ``bounds_supported`` status.
        ``BENCHMARK_OUTPUT_BOUNDS`` can be passed directly here.
    clip_sobol:
        Whether to clip negative Sobol estimates to zero.
    """
    benchmarks = tuple(benchmark_keys) if benchmark_keys is not None else tuple(BENCHMARK_REGISTRY)
    transforms = tuple(transform_keys) if transform_keys is not None else tuple(TRANSFORM_REGISTRY)

    # Merge benchmark_support into the per-pair lookup, with explicit
    # support_by_pair entries taking precedence.
    supports: dict[tuple[str, str], tuple[float, float]] = {}
    if benchmark_support:
        for bk in benchmarks:
            if bk in benchmark_support:
                for tk in transforms:
                    supports[(bk, tk)] = benchmark_support[bk]
    if support_by_pair:
        supports.update(support_by_pair)

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
