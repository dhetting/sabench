from __future__ import annotations

import importlib
import json
from pathlib import Path

import sabench
from sabench.benchmarks import BENCHMARK_REGISTRY
from sabench.metadata.exports import (
    BENCHMARKS_REGISTRY_EXPORT_FILENAME,
    TRANSFORMS_REGISTRY_EXPORT_FILENAME,
    export_registered_benchmark_metadata,
    export_registered_transform_metadata,
)
from sabench.transforms.catalog import TRANSFORMS


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
    assert exported["soft_threshold"]["module"] == "sabench.transforms.nonlinear"
    assert exported["hard_threshold"]["module"] == "sabench.transforms.nonlinear"
    assert exported["ramp"]["module"] == "sabench.transforms.nonlinear"
    assert exported["spike_gaussian"]["module"] == "sabench.transforms.nonlinear"
    assert exported["breakpoint"]["module"] == "sabench.transforms.nonlinear"
    assert exported["hockey_stick"]["module"] == "sabench.transforms.nonlinear"
    assert exported["deadzone"]["module"] == "sabench.transforms.nonlinear"
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
    assert exported["anscombe"]["module"] == "sabench.transforms.statistical"
    assert exported["freeman_tukey"]["module"] == "sabench.transforms.statistical"
    assert exported["asinh_vst"]["module"] == "sabench.transforms.statistical"
    assert exported["modulus_lam05"]["module"] == "sabench.transforms.statistical"
    assert exported["dual_power_lam03"]["module"] == "sabench.transforms.statistical"
    assert exported["yeo_johnson"]["module"] == "sabench.transforms.statistical"
    assert exported["poly4"]["module"] == "sabench.transforms.mathematical"
    assert exported["poly5"]["module"] == "sabench.transforms.mathematical"
    assert exported["poly6"]["module"] == "sabench.transforms.mathematical"
    assert exported["legendre_p3"]["module"] == "sabench.transforms.mathematical"
    assert exported["chebyshev_t4"]["module"] == "sabench.transforms.mathematical"
    assert exported["hermite_he2"]["module"] == "sabench.transforms.mathematical"
    assert exported["hermite_he3"]["module"] == "sabench.transforms.mathematical"
    assert exported["neg_square"]["module"] == "sabench.transforms.mathematical"
    assert exported["smooth_bump"]["module"] == "sabench.transforms.mathematical"
    assert exported["rational_quadratic"]["module"] == "sabench.transforms.mathematical"
    assert exported["inverse_pointwise"]["module"] == "sabench.transforms.mathematical"
    assert exported["atan2pi"]["module"] == "sabench.transforms.mathematical"
    assert exported["exp_neg_sq"]["module"] == "sabench.transforms.mathematical"
    assert exported["exp_pos_sq"]["module"] == "sabench.transforms.mathematical"
    assert exported["inverse_sq"]["module"] == "sabench.transforms.mathematical"
    assert exported["power_exp"]["module"] == "sabench.transforms.mathematical"
    assert exported["triangle_wave"]["module"] == "sabench.transforms.mathematical"
    assert exported["signed_power_p15"]["module"] == "sabench.transforms.mathematical"
    assert exported["signed_power_p05"]["module"] == "sabench.transforms.mathematical"
    assert exported["bernstein_b3"]["module"] == "sabench.transforms.mathematical"
    assert exported["bimodal_flip"]["module"] == "sabench.transforms.mathematical"
    assert exported["donut"]["module"] == "sabench.transforms.mathematical"
    assert exported["var_q95"]["module"] == "sabench.transforms.financial"
    assert exported["cvar_q95"]["module"] == "sabench.transforms.financial"
    assert exported["sharpe_proxy"]["module"] == "sabench.transforms.financial"
    assert exported["drawdown"]["module"] == "sabench.transforms.financial"
    assert exported["fold_change"]["module"] == "sabench.transforms.financial"
    assert exported["excess_return"]["module"] == "sabench.transforms.financial"
    assert exported["hellinger"]["module"] == "sabench.transforms.ecological"
    assert exported["chord_normalise"]["module"] == "sabench.transforms.ecological"
    assert exported["relative_abundance"]["module"] == "sabench.transforms.ecological"
    assert exported["log_ratio"]["module"] == "sabench.transforms.ecological"
    assert exported["weibull_reliability"]["module"] == "sabench.transforms.engineering"
    assert exported["carnot_quadratic"]["module"] == "sabench.transforms.engineering"
    assert exported["arrhenius"]["module"] == "sabench.transforms.engineering"
    assert exported["normalised_stress"]["module"] == "sabench.transforms.engineering"
    assert exported["sigmoid_dose"]["module"] == "sabench.transforms.pharmacological"
    assert exported["hill_response"]["module"] == "sabench.transforms.pharmacological"
    assert exported["log_auc"]["module"] == "sabench.transforms.pharmacological"
    assert exported["emax_model"]["module"] == "sabench.transforms.pharmacological"
    assert exported["fatigue_miner"]["module"] == "sabench.transforms.engineering"
    assert exported["rankine_failure"]["module"] == "sabench.transforms.engineering"
    assert exported["von_mises_stress"]["module"] == "sabench.transforms.engineering"
    assert exported["safety_factor"]["module"] == "sabench.transforms.engineering"
    assert exported["cumulative_damage"]["module"] == "sabench.transforms.engineering"
    assert exported["stress_life"]["module"] == "sabench.transforms.engineering"
    assert exported["log_shift"]["module"] == "sabench.transforms.environmental"
    assert exported["power_law_beta2"]["module"] == "sabench.transforms.environmental"
    assert exported["power_law_beta05"]["module"] == "sabench.transforms.environmental"
    assert exported["box_cox_sqrt"]["module"] == "sabench.transforms.environmental"
    assert exported["box_cox_log"]["module"] == "sabench.transforms.environmental"
    assert exported["clipped_excess_q90"]["module"] == "sabench.transforms.environmental"
    assert exported["exceedance_q75"]["module"] == "sabench.transforms.environmental"
    assert exported["exceedance_q90"]["module"] == "sabench.transforms.environmental"
    assert exported["exceedance_q95"]["module"] == "sabench.transforms.environmental"
    assert exported["exceedance_q99"]["module"] == "sabench.transforms.environmental"
    assert exported["log2_shift"]["module"] == "sabench.transforms.environmental"
    assert exported["log10_shift"]["module"] == "sabench.transforms.environmental"
    assert exported["log_log"]["module"] == "sabench.transforms.environmental"
    assert exported["anomaly_pct"]["module"] == "sabench.transforms.environmental"
    assert exported["bias_correction"]["module"] == "sabench.transforms.environmental"
    assert exported["quantile_delta"]["module"] == "sabench.transforms.environmental"
    assert exported["growing_degree_days"]["module"] == "sabench.transforms.environmental"
    assert exported["std_precip_idx"]["module"] == "sabench.transforms.environmental"
    assert exported["nash_sutcliffe"]["module"] == "sabench.transforms.environmental"
    assert exported["pot_log"]["module"] == "sabench.transforms.environmental"
    assert exported["log_flow"]["module"] == "sabench.transforms.environmental"
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
    assert exported["temporal_log_cumsum"]["module"] == "sabench.transforms.samplewise"
    assert exported["temporal_exceedance_duration"]["module"] == "sabench.transforms.samplewise"
    assert exported["temporal_envelope"]["module"] == "sabench.transforms.samplewise"
    assert exported["temporal_bandpass"]["module"] == "sabench.transforms.samplewise"
    assert exported["temporal_block_avg"]["module"] == "sabench.transforms.aggregation"
    assert exported["gradient_magnitude"]["supported_output_kinds"] == ["spatial"]
    assert exported["regional_mean"]["module"] == "sabench.transforms.aggregation"
    assert exported["block_2x2"]["module"] == "sabench.transforms.aggregation"
    assert exported["block_4x4"]["module"] == "sabench.transforms.aggregation"
    assert exported["block_8x8"]["module"] == "sabench.transforms.aggregation"
    assert exported["exceedance_area"]["module"] == "sabench.transforms.aggregation"
    assert exported["matern_smooth"]["module"] == "sabench.transforms.field_ops"
    assert exported["laplacian_roughness"]["module"] == "sabench.transforms.field_ops"
    assert exported["contour_exceedance"]["module"] == "sabench.transforms.field_ops"
    assert exported["isoline_length"]["module"] == "sabench.transforms.field_ops"


def _metadata_str(metadata: dict[str, object], field_name: str) -> str:
    value = metadata[field_name]
    assert isinstance(value, str)
    return value


def test_benchmark_registry_metadata_points_to_registered_classes() -> None:
    exported = export_registered_benchmark_metadata()

    assert set(exported) == set(BENCHMARK_REGISTRY)
    for name, metadata in exported.items():
        module_name = _metadata_str(metadata, "module")
        class_name = _metadata_str(metadata, "class_name")

        module = importlib.import_module(module_name)
        exported_class = getattr(module, class_name)

        display_name = _metadata_str(metadata, "name")

        assert display_name
        assert class_name == exported_class.__name__
        assert _metadata_str(metadata, "module_name") == module_name.rsplit(".", maxsplit=1)[-1]
        assert exported_class is BENCHMARK_REGISTRY[name].benchmark_cls


def test_transform_registry_metadata_points_to_catalog_functions() -> None:
    exported = export_registered_transform_metadata()

    assert set(exported) == set(TRANSFORMS)
    for key, metadata in exported.items():
        module_name = _metadata_str(metadata, "module")
        function_name = _metadata_str(metadata, "function_name")

        assert _metadata_str(metadata, "key") == key
        assert module_name.startswith("sabench.transforms.")
        assert module_name != "sabench.transforms.transforms"

        module = importlib.import_module(module_name)
        exported_function = getattr(module, function_name)

        assert exported_function is TRANSFORMS[key]["fn"]


def test_committed_registry_metadata_has_no_legacy_transform_monolith_references() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    metadata_root = package_root / "metadata"

    for filename in [BENCHMARKS_REGISTRY_EXPORT_FILENAME, TRANSFORMS_REGISTRY_EXPORT_FILENAME]:
        snapshot = (metadata_root / filename).read_text(encoding="utf-8")
        assert "sabench.transforms.transforms" not in snapshot
