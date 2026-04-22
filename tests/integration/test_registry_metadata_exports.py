from __future__ import annotations

import json
from pathlib import Path

import sabench
from sabench.metadata.exports import (
    BENCHMARKS_REGISTRY_EXPORT_FILENAME,
    TRANSFORMS_REGISTRY_EXPORT_FILENAME,
    export_registered_benchmark_metadata,
    export_registered_transform_metadata,
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def test_registry_metadata_export_files_exist() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    metadata_root = package_root / "metadata"

    assert (metadata_root / BENCHMARKS_REGISTRY_EXPORT_FILENAME).exists()
    assert (metadata_root / TRANSFORMS_REGISTRY_EXPORT_FILENAME).exists()


def test_benchmark_registry_export_file_matches_canonical_export() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    metadata_root = package_root / "metadata"

    exported = export_registered_benchmark_metadata()
    on_disk = _load_json(metadata_root / BENCHMARKS_REGISTRY_EXPORT_FILENAME)

    assert on_disk == exported


def test_transform_registry_export_file_matches_canonical_export() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    metadata_root = package_root / "metadata"

    exported = export_registered_transform_metadata()
    on_disk = _load_json(metadata_root / TRANSFORMS_REGISTRY_EXPORT_FILENAME)

    assert on_disk == exported


def test_benchmark_registry_export_uses_canonical_fields() -> None:
    exported = export_registered_benchmark_metadata()

    assert exported["Ishigami"]["module"] == "sabench.benchmarks.scalar.ishigami"
    assert exported["Ishigami"]["output_kind"] == "scalar"
    assert exported["Campbell2D"]["family"] == "spatial"
    assert exported["Lorenz96"]["has_analytical_st"] is False


def test_transform_registry_export_uses_canonical_fields() -> None:
    exported = export_registered_transform_metadata()

    assert exported["affine_a2_b1"]["module"] == "sabench.transforms.linear"
    assert exported["softplus_b01"]["module"] == "sabench.transforms.nonlinear"
    assert exported["cosh_pointwise"]["module"] == "sabench.transforms.nonlinear"
    assert exported["cbrt_pointwise"]["module"] == "sabench.transforms.nonlinear"
    assert exported["logistic_pointwise"]["module"] == "sabench.transforms.nonlinear"
    assert exported["arctan_pointwise"]["module"] == "sabench.transforms.nonlinear"
    assert exported["sinh_pointwise"]["module"] == "sabench.transforms.nonlinear"
    assert exported["gompertz_cdf"]["module"] == "sabench.transforms.nonlinear"
    assert exported["algebraic_sigmoid"]["module"] == "sabench.transforms.nonlinear"
    assert exported["swish"]["module"] == "sabench.transforms.nonlinear"
    assert exported["mish"]["module"] == "sabench.transforms.nonlinear"
    assert exported["selu"]["module"] == "sabench.transforms.nonlinear"
    assert exported["softsign"]["module"] == "sabench.transforms.nonlinear"
    assert exported["bent_identity"]["module"] == "sabench.transforms.nonlinear"
    assert exported["hard_sigmoid"]["module"] == "sabench.transforms.nonlinear"
    assert exported["hard_tanh"]["module"] == "sabench.transforms.nonlinear"
    assert exported["sample_variance"]["module"] == "sabench.transforms.aggregation"
    assert exported["negentropy_proxy"]["module"] == "sabench.transforms.aggregation"
    assert exported["wasserstein_proxy"]["module"] == "sabench.transforms.aggregation"
    assert exported["energy_distance"]["module"] == "sabench.transforms.aggregation"
    assert exported["renyi_entropy_a2"]["module"] == "sabench.transforms.aggregation"
    assert exported["temporal_rms"]["module"] == "sabench.transforms.aggregation"
    assert exported["temporal_range"]["module"] == "sabench.transforms.aggregation"
    assert exported["temporal_autocorr"]["module"] == "sabench.transforms.aggregation"
    assert exported["temporal_quantile_q10"]["module"] == "sabench.transforms.aggregation"
    assert exported["temporal_quantile_q50"]["module"] == "sabench.transforms.aggregation"
    assert exported["temporal_quantile_q90"]["module"] == "sabench.transforms.aggregation"
    assert exported["rank_transform"]["module"] == "sabench.transforms.statistical"
    assert exported["standardised_anomaly"]["module"] == "sabench.transforms.statistical"
    assert exported["entropy_proxy"]["module"] == "sabench.transforms.statistical"
    assert exported["softmax_shift"]["module"] == "sabench.transforms.statistical"
    assert exported["min_max_normalise"]["module"] == "sabench.transforms.statistical"
    assert exported["robust_scale"]["module"] == "sabench.transforms.statistical"
    assert exported["clamp_sigma"]["module"] == "sabench.transforms.statistical"
    assert exported["quantile_transform"]["module"] == "sabench.transforms.statistical"
    assert exported["winsorise_q10_q90"]["module"] == "sabench.transforms.statistical"
    assert exported["inverse_normal"]["module"] == "sabench.transforms.statistical"
    assert exported["gumbel_cdf"]["module"] == "sabench.transforms.statistical"
    assert exported["frechet_cdf"]["module"] == "sabench.transforms.statistical"
    assert exported["log_normal_cdf"]["module"] == "sabench.transforms.statistical"
    assert exported["return_period"]["module"] == "sabench.transforms.statistical"
    assert exported["johnson_su"]["module"] == "sabench.transforms.statistical"
    assert exported["gev_cdf"]["module"] == "sabench.transforms.statistical"
    assert exported["pareto_tail"]["module"] == "sabench.transforms.statistical"
    assert exported["log_logistic_cdf"]["module"] == "sabench.transforms.statistical"
    assert exported["weibull_reliability"]["module"] == "sabench.transforms.engineering"
    assert exported["fatigue_miner"]["module"] == "sabench.transforms.engineering"
    assert exported["rankine_failure"]["module"] == "sabench.transforms.engineering"
    assert exported["von_mises_stress"]["module"] == "sabench.transforms.engineering"
    assert exported["safety_factor"]["module"] == "sabench.transforms.engineering"
    assert exported["cumulative_damage"]["module"] == "sabench.transforms.engineering"
    assert exported["stress_life"]["module"] == "sabench.transforms.engineering"
    assert exported["cube_pointwise"]["module"] == "sabench.transforms.pointwise"
    assert exported["erf_pointwise"]["module"] == "sabench.transforms.pointwise"
    assert exported["sin_pointwise"]["module"] == "sabench.transforms.pointwise"
    assert exported["cos_pointwise"]["module"] == "sabench.transforms.pointwise"
    assert exported["step_pointwise"]["module"] == "sabench.transforms.pointwise"
    assert exported["log_abs_pointwise"]["module"] == "sabench.transforms.pointwise"
    assert exported["sinc"]["module"] == "sabench.transforms.pointwise"
    assert exported["sin_squared"]["module"] == "sabench.transforms.pointwise"
    assert exported["cos_squared"]["module"] == "sabench.transforms.pointwise"
    assert exported["damped_sin"]["module"] == "sabench.transforms.pointwise"
    assert exported["sawtooth"]["module"] == "sabench.transforms.pointwise"
    assert exported["square_wave"]["module"] == "sabench.transforms.pointwise"
    assert exported["double_sin"]["module"] == "sabench.transforms.pointwise"
    assert exported["sin_cos_product"]["module"] == "sabench.transforms.pointwise"
    assert exported["temporal_cumsum"]["mechanism"] == "samplewise"
    assert exported["gradient_magnitude"]["supported_output_kinds"] == ["spatial"]
