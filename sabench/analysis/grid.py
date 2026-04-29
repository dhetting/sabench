"""Registry-driven benchmark/transform grid execution utilities.

This module provides the reusable execution layer for noncommutativity grid
analyses. It deliberately keeps notebook-specific rendering and file export out
of the runtime package: notebooks can call these functions to classify pairs,
evaluate compatible benchmark/transform combinations, and obtain one tidy row
per pair.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from sabench.analysis.estimators import jansen_s1_st
from sabench.analysis.noncommutativity import sobol_profile_shift_metrics
from sabench.benchmarks.registry import BENCHMARK_REGISTRY, get_benchmark_definition
from sabench.benchmarks.types import BenchmarkOutputKind
from sabench.transforms.registry import TRANSFORM_REGISTRY, get_transform_definition
from sabench.transforms.types import TransformMechanism, TransformOutputKind, TransformTag

PairStatus = Literal["included", "excluded"]
MetricsStatus = Literal[
    "computed",
    "not_applicable",
    "failed_raw_evaluation",
    "failed_transform_evaluation",
    "failed_nonfinite_raw_output",
    "failed_nonfinite_transformed_output",
    "failed_output_shape",
    "failed_sobol_estimation",
    "failed_metric_computation",
]


@dataclass(frozen=True, slots=True)
class PairCompatibility:
    """Static compatibility classification for one benchmark/transform pair."""

    benchmark_key: str
    transform_key: str
    benchmark_output_kind: BenchmarkOutputKind
    transform_mechanism: TransformMechanism
    transform_supported_output_kinds: tuple[TransformOutputKind, ...]
    transform_tags: tuple[TransformTag, ...]
    pair_status: PairStatus
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return a tabular representation of the compatibility classification."""
        values = asdict(self)
        values["transform_supported_output_kinds"] = ";".join(self.transform_supported_output_kinds)
        values["transform_tags"] = ";".join(self.transform_tags)
        return values


@dataclass(frozen=True, slots=True)
class NoncommutativityGridResult:
    """Result row for one noncommutativity benchmark/transform evaluation."""

    benchmark_key: str
    transform_key: str
    pair_status: PairStatus
    metrics_status: MetricsStatus
    reason: str
    benchmark_output_kind: BenchmarkOutputKind
    transform_mechanism: TransformMechanism
    transform_tags: tuple[TransformTag, ...]
    n_base: int
    seed: int | None
    n_inputs: int
    n_evaluations: int | None = None
    raw_output_shape: tuple[int, ...] | None = None
    transformed_output_shape: tuple[int, ...] | None = None
    raw_output_finite: bool | None = None
    transformed_output_finite: bool | None = None
    raw_variance: float | None = None
    transformed_variance: float | None = None
    metrics: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a flat dictionary suitable for a notebook result table."""
        values = asdict(self)
        metrics = values.pop("metrics") or {}
        values["transform_tags"] = ";".join(self.transform_tags)
        values["raw_output_shape"] = _shape_to_string(self.raw_output_shape)
        values["transformed_output_shape"] = _shape_to_string(self.transformed_output_shape)
        values.update(metrics)
        return values


def classify_noncommutativity_pair(
    benchmark_key: str,
    transform_key: str,
) -> PairCompatibility:
    """Classify whether a benchmark/transform pair is statically compatible."""
    benchmark_definition = get_benchmark_definition(benchmark_key)
    transform_definition = get_transform_definition(transform_key)
    benchmark_output_kind = benchmark_definition.spec.output_kind
    transform_spec = transform_definition.spec

    if benchmark_output_kind not in transform_spec.supported_output_kinds:
        supported = ", ".join(transform_spec.supported_output_kinds)
        return PairCompatibility(
            benchmark_key=benchmark_key,
            transform_key=transform_key,
            benchmark_output_kind=benchmark_output_kind,
            transform_mechanism=transform_spec.mechanism,
            transform_supported_output_kinds=transform_spec.supported_output_kinds,
            transform_tags=transform_spec.tags,
            pair_status="excluded",
            reason=(
                f"benchmark output kind {benchmark_output_kind!r} is not supported; "
                f"transform supports {supported}"
            ),
        )

    return PairCompatibility(
        benchmark_key=benchmark_key,
        transform_key=transform_key,
        benchmark_output_kind=benchmark_output_kind,
        transform_mechanism=transform_spec.mechanism,
        transform_supported_output_kinds=transform_spec.supported_output_kinds,
        transform_tags=transform_spec.tags,
        pair_status="included",
        reason="compatible output kind",
    )


def evaluate_noncommutativity_pair(
    benchmark_key: str,
    transform_key: str,
    *,
    n_base: int = 512,
    seed: int | None = 0,
    tau: float = 0.05,
    top_k: int = 3,
    clip_sobol: bool = True,
) -> NoncommutativityGridResult:
    """Evaluate one benchmark/transform pair and compute profile-shift metrics.

    The function is intentionally failure-tolerant: incompatible or numerically
    invalid pairs return a result row with a failure status and reason rather
    than raising. Unknown benchmark or transform keys still raise through the
    canonical registries because they are caller errors.
    """
    _validate_n_base(n_base)
    compatibility = classify_noncommutativity_pair(benchmark_key, transform_key)
    benchmark_definition = get_benchmark_definition(benchmark_key)
    transform_definition = get_transform_definition(transform_key)
    benchmark = benchmark_definition.benchmark_cls()

    if compatibility.pair_status == "excluded":
        return _result_from_compatibility(
            compatibility,
            metrics_status="not_applicable",
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
        )

    sample = benchmark.sample(n_base, seed=seed)
    try:
        raw_output = np.asarray(benchmark.evaluate(sample), dtype=float)
    except Exception as exc:  # pragma: no cover - defensive row-level failure path
        return _result_from_compatibility(
            compatibility,
            metrics_status="failed_raw_evaluation",
            reason=str(exc),
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
        )

    raw_validation_error = _validate_output(raw_output, expected_rows=sample.shape[0])
    if raw_validation_error is not None:
        status = _output_validation_status(raw_validation_error, raw=True)
        return _result_from_compatibility(
            compatibility,
            metrics_status=status,
            reason=raw_validation_error,
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
            n_evaluations=sample.shape[0],
            raw_output_shape=raw_output.shape,
            raw_output_finite=bool(np.all(np.isfinite(raw_output))),
        )

    try:
        transformed_output = np.asarray(transform_definition.transform(raw_output), dtype=float)
    except Exception as exc:
        return _result_from_compatibility(
            compatibility,
            metrics_status="failed_transform_evaluation",
            reason=str(exc),
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
            n_evaluations=sample.shape[0],
            raw_output_shape=raw_output.shape,
            raw_output_finite=True,
            raw_variance=_total_variance(raw_output),
        )

    transformed_validation_error = _validate_output(
        transformed_output,
        expected_rows=sample.shape[0],
    )
    if transformed_validation_error is not None:
        status = _output_validation_status(transformed_validation_error, raw=False)
        return _result_from_compatibility(
            compatibility,
            metrics_status=status,
            reason=transformed_validation_error,
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
            n_evaluations=sample.shape[0],
            raw_output_shape=raw_output.shape,
            transformed_output_shape=transformed_output.shape,
            raw_output_finite=True,
            transformed_output_finite=bool(np.all(np.isfinite(transformed_output))),
            raw_variance=_total_variance(raw_output),
        )

    raw_matrix = as_estimator_output(raw_output)
    transformed_matrix = as_estimator_output(transformed_output)

    try:
        raw_s1, raw_st = jansen_s1_st(raw_matrix, N=n_base, d=benchmark.d, clip=clip_sobol)
        transformed_s1, transformed_st = jansen_s1_st(
            transformed_matrix,
            N=n_base,
            d=benchmark.d,
            clip=clip_sobol,
        )
        raw_s1_profile = variance_weighted_sobol_profile(raw_s1, raw_matrix)
        raw_st_profile = variance_weighted_sobol_profile(raw_st, raw_matrix)
        transformed_s1_profile = variance_weighted_sobol_profile(
            transformed_s1,
            transformed_matrix,
        )
        transformed_st_profile = variance_weighted_sobol_profile(
            transformed_st,
            transformed_matrix,
        )
    except Exception as exc:  # pragma: no cover - defensive row-level failure path
        return _result_from_compatibility(
            compatibility,
            metrics_status="failed_sobol_estimation",
            reason=str(exc),
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
            n_evaluations=sample.shape[0],
            raw_output_shape=raw_output.shape,
            transformed_output_shape=transformed_output.shape,
            raw_output_finite=True,
            transformed_output_finite=True,
            raw_variance=_total_variance(raw_output),
            transformed_variance=_total_variance(transformed_output),
        )

    try:
        metrics = sobol_profile_shift_metrics(
            raw_s1_profile,
            transformed_s1_profile,
            raw_st_profile,
            transformed_st_profile,
            tau=tau,
            top_k=top_k,
        )
    except Exception as exc:  # pragma: no cover - defensive row-level failure path
        return _result_from_compatibility(
            compatibility,
            metrics_status="failed_metric_computation",
            reason=str(exc),
            n_base=n_base,
            seed=seed,
            n_inputs=benchmark.d,
            n_evaluations=sample.shape[0],
            raw_output_shape=raw_output.shape,
            transformed_output_shape=transformed_output.shape,
            raw_output_finite=True,
            transformed_output_finite=True,
            raw_variance=_total_variance(raw_output),
            transformed_variance=_total_variance(transformed_output),
        )

    return _result_from_compatibility(
        compatibility,
        metrics_status="computed",
        reason="metrics computed",
        n_base=n_base,
        seed=seed,
        n_inputs=benchmark.d,
        n_evaluations=sample.shape[0],
        raw_output_shape=raw_output.shape,
        transformed_output_shape=transformed_output.shape,
        raw_output_finite=True,
        transformed_output_finite=True,
        raw_variance=_total_variance(raw_output),
        transformed_variance=_total_variance(transformed_output),
        metrics=metrics,
    )


def evaluate_noncommutativity_grid(
    benchmark_keys: Iterable[str] | None = None,
    transform_keys: Iterable[str] | None = None,
    *,
    n_base: int = 512,
    seed: int | None = 0,
    tau: float = 0.05,
    top_k: int = 3,
    clip_sobol: bool = True,
) -> tuple[NoncommutativityGridResult, ...]:
    """Evaluate a deterministic benchmark/transform grid."""
    benchmarks = tuple(benchmark_keys) if benchmark_keys is not None else tuple(BENCHMARK_REGISTRY)
    transforms = tuple(transform_keys) if transform_keys is not None else tuple(TRANSFORM_REGISTRY)
    return tuple(
        evaluate_noncommutativity_pair(
            benchmark_key,
            transform_key,
            n_base=n_base,
            seed=seed,
            tau=tau,
            top_k=top_k,
            clip_sobol=clip_sobol,
        )
        for benchmark_key in benchmarks
        for transform_key in transforms
    )


def as_estimator_output(output: np.ndarray) -> np.ndarray:
    """Return output as a Jansen-estimator-compatible array."""
    values = np.asarray(output, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim < 1:
        raise ValueError("output must include a sample axis")
    return values.reshape(values.shape[0], -1)


def variance_weighted_sobol_profile(indices: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Aggregate scalar or multi-output Sobol indices to one profile per input."""
    index_values = np.asarray(indices, dtype=float)
    output_matrix = as_estimator_output(output)

    if index_values.ndim == 1:
        return np.clip(index_values.copy(), 0.0, 1.0)
    if index_values.ndim != 2:
        raise ValueError("indices must be one- or two-dimensional")
    if index_values.shape[1] != output_matrix.shape[1]:
        raise ValueError(
            "indices output dimension must match flattened output dimension; "
            f"got {index_values.shape[1]} and {output_matrix.shape[1]}"
        )

    output_variance = output_matrix.var(axis=0).ravel()
    total_variance = float(output_variance.sum())
    if total_variance <= 1e-30:
        profile = index_values.mean(axis=1)
    else:
        profile = (index_values * output_variance[None, :]).sum(axis=1) / total_variance
    return np.clip(profile, 0.0, 1.0)


def _result_from_compatibility(
    compatibility: PairCompatibility,
    *,
    metrics_status: MetricsStatus,
    n_base: int,
    seed: int | None,
    n_inputs: int,
    reason: str | None = None,
    n_evaluations: int | None = None,
    raw_output_shape: tuple[int, ...] | None = None,
    transformed_output_shape: tuple[int, ...] | None = None,
    raw_output_finite: bool | None = None,
    transformed_output_finite: bool | None = None,
    raw_variance: float | None = None,
    transformed_variance: float | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> NoncommutativityGridResult:
    return NoncommutativityGridResult(
        benchmark_key=compatibility.benchmark_key,
        transform_key=compatibility.transform_key,
        pair_status=compatibility.pair_status,
        metrics_status=metrics_status,
        reason=compatibility.reason if reason is None else reason,
        benchmark_output_kind=compatibility.benchmark_output_kind,
        transform_mechanism=compatibility.transform_mechanism,
        transform_tags=compatibility.transform_tags,
        n_base=n_base,
        seed=seed,
        n_inputs=n_inputs,
        n_evaluations=n_evaluations,
        raw_output_shape=raw_output_shape,
        transformed_output_shape=transformed_output_shape,
        raw_output_finite=raw_output_finite,
        transformed_output_finite=transformed_output_finite,
        raw_variance=raw_variance,
        transformed_variance=transformed_variance,
        metrics=metrics,
    )


def _validate_n_base(n_base: int) -> None:
    if n_base <= 0:
        raise ValueError("n_base must be positive")


def _validate_output(output: np.ndarray, *, expected_rows: int) -> str | None:
    if output.ndim < 1:
        return "output must include a sample axis"
    if output.shape[0] != expected_rows:
        return f"output sample axis has length {output.shape[0]}, expected {expected_rows}"
    if not np.all(np.isfinite(output)):
        return "output contains nonfinite values"
    return None


def _output_validation_status(reason: str, *, raw: bool) -> MetricsStatus:
    if "nonfinite" in reason:
        return "failed_nonfinite_raw_output" if raw else "failed_nonfinite_transformed_output"
    return "failed_output_shape"


def _total_variance(output: np.ndarray) -> float:
    matrix = as_estimator_output(output)
    return float(matrix.var(axis=0).sum())


def _shape_to_string(shape: tuple[int, ...] | None) -> str | None:
    if shape is None:
        return None
    return "x".join(str(part) for part in shape)
