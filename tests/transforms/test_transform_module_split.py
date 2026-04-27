from __future__ import annotations

from pathlib import Path

import numpy as np

import sabench
from sabench.transforms import TRANSFORMS, apply_transform, get_transform_spec
from sabench.transforms.aggregation import (
    t_block_2x2,
    t_block_4x4,
    t_block_8x8,
    t_energy_distance_proxy,
    t_entropy_renyi,
    t_exceedance_area,
    t_interquartile_range,
    t_negentropy_proxy,
    t_percentile_q10,
    t_percentile_q90,
    t_regional_mean,
    t_sample_kurtosis,
    t_sample_skewness,
    t_sample_variance,
    t_temporal_autocorr,
    t_temporal_peak,
    t_temporal_quantile,
    t_temporal_range,
    t_temporal_rms,
    t_wasserstein_proxy,
)
from sabench.transforms.ecological import (
    t_chord_dist,
    t_hellinger,
    t_log_ratio,
    t_relative_abundance,
)
from sabench.transforms.engineering import (
    t_cumulative_damage,
    t_fatigue_miner,
    t_rankine_failure,
    t_safety_factor,
    t_stress_life,
    t_von_mises,
    t_weibull_reliability,
)
from sabench.transforms.environmental import (
    t_anomaly_pct,
    t_bias_correction,
    t_box_cox,
    t_clipped_excess,
    t_exceed_q75,
    t_exceed_q90,
    t_exceed_q95,
    t_exceed_q99,
    t_growing_degree_days,
    t_log2_shift,
    t_log10_shift,
    t_log_flow,
    t_log_log,
    t_log_shift,
    t_nash_sutcliffe,
    t_pot_log,
    t_power_law,
    t_quantile_delta,
    t_standardised_precip_idx,
)
from sabench.transforms.field_ops import (
    t_contour_exceedance,
    t_gradient_magnitude,
    t_isoline_length,
    t_laplacian_roughness,
    t_matern_smooth,
)
from sabench.transforms.financial import (
    t_cvar,
    t_drawdown,
    t_excess_return,
    t_fold_change,
    t_sharpe_proxy,
    t_var_proxy,
)
from sabench.transforms.linear import t_affine
from sabench.transforms.mathematical import (
    t_atan2pi,
    t_chebyshev_t4,
    t_exp_neg_sq,
    t_exp_pos_sq,
    t_hermite_he2,
    t_hermite_he3,
    t_inverse_abs,
    t_inverse_sq,
    t_legendre_p3,
    t_neg_square,
    t_poly4,
    t_poly5,
    t_poly6,
    t_power_exp,
    t_rational_quadratic,
    t_smooth_bump,
)
from sabench.transforms.nonlinear import (
    t_algebraic_sigmoid,
    t_arctan_pointwise,
    t_bent_identity,
    t_breakpoint,
    t_cbrt_pointwise,
    t_cosh_pointwise,
    t_deadzone,
    t_gompertz,
    t_hard_sigmoid,
    t_hard_tanh,
    t_hard_threshold,
    t_hockey_stick,
    t_logistic_pointwise,
    t_mish,
    t_ramp,
    t_selu,
    t_sinh_pointwise,
    t_soft_threshold,
    t_softplus_pointwise,
    t_softsign,
    t_spike,
    t_swish,
)
from sabench.transforms.pharmacological import (
    t_emax_model,
    t_hill_response,
    t_log_auc,
    t_sigmoid_dose,
)
from sabench.transforms.pointwise import (
    t_abs_pointwise,
    t_cos_pointwise,
    t_cos_squared,
    t_cube_pointwise,
    t_damped_sin,
    t_double_sin,
    t_erf_pointwise,
    t_exp_pointwise,
    t_log1p_abs,
    t_log_abs,
    t_relu_pointwise,
    t_sawtooth,
    t_sin_cos_product,
    t_sin_pointwise,
    t_sin_squared,
    t_sinc,
    t_sqrt_abs,
    t_square_pointwise,
    t_square_wave,
    t_step_pointwise,
    t_tanh_pointwise,
)
from sabench.transforms.samplewise import t_temporal_cumsum
from sabench.transforms.statistical import (
    t_anscombe,
    t_asinh_vs,
    t_clamp_sigma,
    t_dual_power,
    t_entropy_proxy,
    t_frechet_cdf,
    t_freeman_tukey,
    t_gev_cdf,
    t_gumbel_cdf,
    t_inverse_normal,
    t_johnson_su,
    t_log_logistic_cdf,
    t_log_normal_cdf,
    t_min_max_normalise,
    t_modulus,
    t_pareto_tail,
    t_quantile_normalise,
    t_rank_transform,
    t_return_period,
    t_robust_scale,
    t_softmax_shift,
    t_standardised_anomaly,
    t_winsorise,
)


def test_representative_transform_specs_point_to_split_modules() -> None:
    assert get_transform_spec("affine_a2_b1").module == "sabench.transforms.linear"
    assert get_transform_spec("tanh_a03").module == "sabench.transforms.pointwise"
    assert get_transform_spec("square_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("exp_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("relu_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("log1p_positive").module == "sabench.transforms.pointwise"
    assert get_transform_spec("sqrt_abs").module == "sabench.transforms.pointwise"
    assert get_transform_spec("abs_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("cube_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("erf_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("sin_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("cos_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("step_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("log_abs_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("poly4").module == "sabench.transforms.mathematical"
    assert get_transform_spec("poly5").module == "sabench.transforms.mathematical"
    assert get_transform_spec("poly6").module == "sabench.transforms.mathematical"
    assert get_transform_spec("legendre_p3").module == "sabench.transforms.mathematical"
    assert get_transform_spec("chebyshev_t4").module == "sabench.transforms.mathematical"
    assert get_transform_spec("hermite_he2").module == "sabench.transforms.mathematical"
    assert get_transform_spec("hermite_he3").module == "sabench.transforms.mathematical"
    assert get_transform_spec("neg_square").module == "sabench.transforms.mathematical"
    assert get_transform_spec("smooth_bump").module == "sabench.transforms.mathematical"
    assert get_transform_spec("rational_quadratic").module == "sabench.transforms.mathematical"
    assert get_transform_spec("inverse_pointwise").module == "sabench.transforms.mathematical"
    assert get_transform_spec("atan2pi").module == "sabench.transforms.mathematical"
    assert get_transform_spec("exp_neg_sq").module == "sabench.transforms.mathematical"
    assert get_transform_spec("exp_pos_sq").module == "sabench.transforms.mathematical"
    assert get_transform_spec("inverse_sq").module == "sabench.transforms.mathematical"
    assert get_transform_spec("power_exp").module == "sabench.transforms.mathematical"
    assert get_transform_spec("sinc").module == "sabench.transforms.pointwise"
    assert get_transform_spec("sin_squared").module == "sabench.transforms.pointwise"
    assert get_transform_spec("cos_squared").module == "sabench.transforms.pointwise"
    assert get_transform_spec("damped_sin").module == "sabench.transforms.pointwise"
    assert get_transform_spec("sawtooth").module == "sabench.transforms.pointwise"
    assert get_transform_spec("square_wave").module == "sabench.transforms.pointwise"
    assert get_transform_spec("double_sin").module == "sabench.transforms.pointwise"
    assert get_transform_spec("sin_cos_product").module == "sabench.transforms.pointwise"
    assert get_transform_spec("softplus_b01").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("cosh_pointwise").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("cbrt_pointwise").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("logistic_pointwise").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("arctan_pointwise").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("sinh_pointwise").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("gompertz_cdf").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("algebraic_sigmoid").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("swish").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("mish").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("selu").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("softsign").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("bent_identity").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("hard_sigmoid").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("hard_tanh").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("soft_threshold").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("hard_threshold").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("ramp").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("spike_gaussian").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("breakpoint").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("hockey_stick").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("deadzone").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("temporal_cumsum").module == "sabench.transforms.samplewise"
    assert get_transform_spec("temporal_peak").module == "sabench.transforms.aggregation"
    assert get_transform_spec("temporal_rms").module == "sabench.transforms.aggregation"
    assert get_transform_spec("temporal_range").module == "sabench.transforms.aggregation"
    assert get_transform_spec("temporal_autocorr").module == "sabench.transforms.aggregation"
    assert get_transform_spec("temporal_quantile_q10").module == "sabench.transforms.aggregation"
    assert get_transform_spec("temporal_quantile_q50").module == "sabench.transforms.aggregation"
    assert get_transform_spec("temporal_quantile_q90").module == "sabench.transforms.aggregation"
    assert get_transform_spec("sample_variance").module == "sabench.transforms.aggregation"
    assert get_transform_spec("negentropy_proxy").module == "sabench.transforms.aggregation"
    assert get_transform_spec("wasserstein_proxy").module == "sabench.transforms.aggregation"
    assert get_transform_spec("energy_distance").module == "sabench.transforms.aggregation"
    assert get_transform_spec("renyi_entropy_a2").module == "sabench.transforms.aggregation"
    assert get_transform_spec("sample_skewness").module == "sabench.transforms.aggregation"
    assert get_transform_spec("sample_kurtosis").module == "sabench.transforms.aggregation"
    assert get_transform_spec("percentile_q10").module == "sabench.transforms.aggregation"
    assert get_transform_spec("percentile_q90").module == "sabench.transforms.aggregation"
    assert get_transform_spec("iqr").module == "sabench.transforms.aggregation"
    assert get_transform_spec("rank_transform").module == "sabench.transforms.statistical"
    assert get_transform_spec("standardised_anomaly").module == "sabench.transforms.statistical"
    assert get_transform_spec("entropy_proxy").module == "sabench.transforms.statistical"
    assert get_transform_spec("softmax_shift").module == "sabench.transforms.statistical"
    assert get_transform_spec("min_max_normalise").module == "sabench.transforms.statistical"
    assert get_transform_spec("robust_scale").module == "sabench.transforms.statistical"
    assert get_transform_spec("clamp_sigma").module == "sabench.transforms.statistical"
    assert get_transform_spec("quantile_transform").module == "sabench.transforms.statistical"
    assert get_transform_spec("winsorise_q10_q90").module == "sabench.transforms.statistical"
    assert get_transform_spec("inverse_normal").module == "sabench.transforms.statistical"
    assert get_transform_spec("gumbel_cdf").module == "sabench.transforms.statistical"
    assert get_transform_spec("frechet_cdf").module == "sabench.transforms.statistical"
    assert get_transform_spec("log_normal_cdf").module == "sabench.transforms.statistical"
    assert get_transform_spec("return_period").module == "sabench.transforms.statistical"
    assert get_transform_spec("johnson_su").module == "sabench.transforms.statistical"
    assert get_transform_spec("gev_cdf").module == "sabench.transforms.statistical"
    assert get_transform_spec("pareto_tail").module == "sabench.transforms.statistical"
    assert get_transform_spec("log_logistic_cdf").module == "sabench.transforms.statistical"
    assert get_transform_spec("anscombe").module == "sabench.transforms.statistical"
    assert get_transform_spec("freeman_tukey").module == "sabench.transforms.statistical"
    assert get_transform_spec("asinh_vst").module == "sabench.transforms.statistical"
    assert get_transform_spec("modulus_lam05").module == "sabench.transforms.statistical"
    assert get_transform_spec("dual_power_lam03").module == "sabench.transforms.statistical"
    assert get_transform_spec("var_q95").module == "sabench.transforms.financial"
    assert get_transform_spec("cvar_q95").module == "sabench.transforms.financial"
    assert get_transform_spec("sharpe_proxy").module == "sabench.transforms.financial"
    assert get_transform_spec("drawdown").module == "sabench.transforms.financial"
    assert get_transform_spec("fold_change").module == "sabench.transforms.financial"
    assert get_transform_spec("excess_return").module == "sabench.transforms.financial"
    assert get_transform_spec("hellinger").module == "sabench.transforms.ecological"
    assert get_transform_spec("chord_normalise").module == "sabench.transforms.ecological"
    assert get_transform_spec("relative_abundance").module == "sabench.transforms.ecological"
    assert get_transform_spec("log_ratio").module == "sabench.transforms.ecological"
    assert get_transform_spec("weibull_reliability").module == "sabench.transforms.engineering"
    assert get_transform_spec("fatigue_miner").module == "sabench.transforms.engineering"
    assert get_transform_spec("rankine_failure").module == "sabench.transforms.engineering"
    assert get_transform_spec("von_mises_stress").module == "sabench.transforms.engineering"
    assert get_transform_spec("safety_factor").module == "sabench.transforms.engineering"
    assert get_transform_spec("cumulative_damage").module == "sabench.transforms.engineering"
    assert get_transform_spec("stress_life").module == "sabench.transforms.engineering"
    assert get_transform_spec("sigmoid_dose").module == "sabench.transforms.pharmacological"
    assert get_transform_spec("hill_response").module == "sabench.transforms.pharmacological"
    assert get_transform_spec("log_auc").module == "sabench.transforms.pharmacological"
    assert get_transform_spec("emax_model").module == "sabench.transforms.pharmacological"
    assert get_transform_spec("log_shift").module == "sabench.transforms.environmental"
    assert get_transform_spec("power_law_beta2").module == "sabench.transforms.environmental"
    assert get_transform_spec("power_law_beta05").module == "sabench.transforms.environmental"
    assert get_transform_spec("box_cox_sqrt").module == "sabench.transforms.environmental"
    assert get_transform_spec("box_cox_log").module == "sabench.transforms.environmental"
    assert get_transform_spec("clipped_excess_q90").module == "sabench.transforms.environmental"
    assert get_transform_spec("exceedance_q75").module == "sabench.transforms.environmental"
    assert get_transform_spec("exceedance_q90").module == "sabench.transforms.environmental"
    assert get_transform_spec("exceedance_q95").module == "sabench.transforms.environmental"
    assert get_transform_spec("exceedance_q99").module == "sabench.transforms.environmental"
    assert get_transform_spec("log2_shift").module == "sabench.transforms.environmental"
    assert get_transform_spec("log10_shift").module == "sabench.transforms.environmental"
    assert get_transform_spec("log_log").module == "sabench.transforms.environmental"
    assert get_transform_spec("anomaly_pct").module == "sabench.transforms.environmental"
    assert get_transform_spec("bias_correction").module == "sabench.transforms.environmental"
    assert get_transform_spec("quantile_delta").module == "sabench.transforms.environmental"
    assert get_transform_spec("growing_degree_days").module == "sabench.transforms.environmental"
    assert get_transform_spec("std_precip_idx").module == "sabench.transforms.environmental"
    assert get_transform_spec("nash_sutcliffe").module == "sabench.transforms.environmental"
    assert get_transform_spec("pot_log").module == "sabench.transforms.environmental"
    assert get_transform_spec("log_flow").module == "sabench.transforms.environmental"
    assert get_transform_spec("gradient_magnitude").module == "sabench.transforms.field_ops"
    assert get_transform_spec("regional_mean").module == "sabench.transforms.aggregation"
    assert get_transform_spec("block_2x2").module == "sabench.transforms.aggregation"
    assert get_transform_spec("block_4x4").module == "sabench.transforms.aggregation"
    assert get_transform_spec("block_8x8").module == "sabench.transforms.aggregation"
    assert get_transform_spec("exceedance_area").module == "sabench.transforms.aggregation"
    assert get_transform_spec("matern_smooth").module == "sabench.transforms.field_ops"
    assert get_transform_spec("laplacian_roughness").module == "sabench.transforms.field_ops"
    assert get_transform_spec("contour_exceedance").module == "sabench.transforms.field_ops"
    assert get_transform_spec("isoline_length").module == "sabench.transforms.field_ops"


def test_legacy_transform_registry_uses_split_module_functions() -> None:
    assert TRANSFORMS["affine_a2_b1"]["fn"] is t_affine
    assert TRANSFORMS["tanh_a03"]["fn"] is t_tanh_pointwise
    assert TRANSFORMS["square_pointwise"]["fn"] is t_square_pointwise
    assert TRANSFORMS["exp_pointwise"]["fn"] is t_exp_pointwise
    assert TRANSFORMS["relu_pointwise"]["fn"] is t_relu_pointwise
    assert TRANSFORMS["log1p_positive"]["fn"] is t_log1p_abs
    assert TRANSFORMS["sqrt_abs"]["fn"] is t_sqrt_abs
    assert TRANSFORMS["abs_pointwise"]["fn"] is t_abs_pointwise
    assert TRANSFORMS["cube_pointwise"]["fn"] is t_cube_pointwise
    assert TRANSFORMS["erf_pointwise"]["fn"] is t_erf_pointwise
    assert TRANSFORMS["sin_pointwise"]["fn"] is t_sin_pointwise
    assert TRANSFORMS["cos_pointwise"]["fn"] is t_cos_pointwise
    assert TRANSFORMS["step_pointwise"]["fn"] is t_step_pointwise
    assert TRANSFORMS["log_abs_pointwise"]["fn"] is t_log_abs
    assert TRANSFORMS["neg_square"]["fn"] is t_neg_square
    assert TRANSFORMS["smooth_bump"]["fn"] is t_smooth_bump
    assert TRANSFORMS["rational_quadratic"]["fn"] is t_rational_quadratic
    assert TRANSFORMS["inverse_pointwise"]["fn"] is t_inverse_abs
    assert TRANSFORMS["atan2pi"]["fn"] is t_atan2pi
    assert TRANSFORMS["exp_neg_sq"]["fn"] is t_exp_neg_sq
    assert TRANSFORMS["exp_pos_sq"]["fn"] is t_exp_pos_sq
    assert TRANSFORMS["inverse_sq"]["fn"] is t_inverse_sq
    assert TRANSFORMS["power_exp"]["fn"] is t_power_exp
    assert TRANSFORMS["sinc"]["fn"] is t_sinc
    assert TRANSFORMS["sin_squared"]["fn"] is t_sin_squared
    assert TRANSFORMS["cos_squared"]["fn"] is t_cos_squared
    assert TRANSFORMS["damped_sin"]["fn"] is t_damped_sin
    assert TRANSFORMS["sawtooth"]["fn"] is t_sawtooth
    assert TRANSFORMS["square_wave"]["fn"] is t_square_wave
    assert TRANSFORMS["double_sin"]["fn"] is t_double_sin
    assert TRANSFORMS["sin_cos_product"]["fn"] is t_sin_cos_product
    assert TRANSFORMS["softplus_b01"]["fn"] is t_softplus_pointwise
    assert TRANSFORMS["cosh_pointwise"]["fn"] is t_cosh_pointwise
    assert TRANSFORMS["cbrt_pointwise"]["fn"] is t_cbrt_pointwise
    assert TRANSFORMS["logistic_pointwise"]["fn"] is t_logistic_pointwise
    assert TRANSFORMS["arctan_pointwise"]["fn"] is t_arctan_pointwise
    assert TRANSFORMS["sinh_pointwise"]["fn"] is t_sinh_pointwise
    assert TRANSFORMS["gompertz_cdf"]["fn"] is t_gompertz
    assert TRANSFORMS["algebraic_sigmoid"]["fn"] is t_algebraic_sigmoid
    assert TRANSFORMS["swish"]["fn"] is t_swish
    assert TRANSFORMS["mish"]["fn"] is t_mish
    assert TRANSFORMS["selu"]["fn"] is t_selu
    assert TRANSFORMS["softsign"]["fn"] is t_softsign
    assert TRANSFORMS["bent_identity"]["fn"] is t_bent_identity
    assert TRANSFORMS["hard_sigmoid"]["fn"] is t_hard_sigmoid
    assert TRANSFORMS["hard_tanh"]["fn"] is t_hard_tanh
    assert TRANSFORMS["soft_threshold"]["fn"] is t_soft_threshold
    assert TRANSFORMS["hard_threshold"]["fn"] is t_hard_threshold
    assert TRANSFORMS["ramp"]["fn"] is t_ramp
    assert TRANSFORMS["spike_gaussian"]["fn"] is t_spike
    assert TRANSFORMS["breakpoint"]["fn"] is t_breakpoint
    assert TRANSFORMS["hockey_stick"]["fn"] is t_hockey_stick
    assert TRANSFORMS["deadzone"]["fn"] is t_deadzone
    assert TRANSFORMS["temporal_cumsum"]["fn"] is t_temporal_cumsum
    assert TRANSFORMS["temporal_peak"]["fn"] is t_temporal_peak
    assert TRANSFORMS["temporal_rms"]["fn"] is t_temporal_rms
    assert TRANSFORMS["temporal_range"]["fn"] is t_temporal_range
    assert TRANSFORMS["temporal_autocorr"]["fn"] is t_temporal_autocorr
    assert TRANSFORMS["temporal_quantile_q10"]["fn"] is t_temporal_quantile
    assert TRANSFORMS["temporal_quantile_q50"]["fn"] is t_temporal_quantile
    assert TRANSFORMS["temporal_quantile_q90"]["fn"] is t_temporal_quantile
    assert TRANSFORMS["sample_variance"]["fn"] is t_sample_variance
    assert TRANSFORMS["negentropy_proxy"]["fn"] is t_negentropy_proxy
    assert TRANSFORMS["wasserstein_proxy"]["fn"] is t_wasserstein_proxy
    assert TRANSFORMS["energy_distance"]["fn"] is t_energy_distance_proxy
    assert TRANSFORMS["renyi_entropy_a2"]["fn"] is t_entropy_renyi
    assert TRANSFORMS["sample_skewness"]["fn"] is t_sample_skewness
    assert TRANSFORMS["sample_kurtosis"]["fn"] is t_sample_kurtosis
    assert TRANSFORMS["percentile_q10"]["fn"] is t_percentile_q10
    assert TRANSFORMS["percentile_q90"]["fn"] is t_percentile_q90
    assert TRANSFORMS["iqr"]["fn"] is t_interquartile_range
    assert TRANSFORMS["rank_transform"]["fn"] is t_rank_transform
    assert TRANSFORMS["standardised_anomaly"]["fn"] is t_standardised_anomaly
    assert TRANSFORMS["entropy_proxy"]["fn"] is t_entropy_proxy
    assert TRANSFORMS["softmax_shift"]["fn"] is t_softmax_shift
    assert TRANSFORMS["min_max_normalise"]["fn"] is t_min_max_normalise
    assert TRANSFORMS["robust_scale"]["fn"] is t_robust_scale
    assert TRANSFORMS["clamp_sigma"]["fn"] is t_clamp_sigma
    assert TRANSFORMS["quantile_transform"]["fn"] is t_quantile_normalise
    assert TRANSFORMS["winsorise_q10_q90"]["fn"] is t_winsorise
    assert TRANSFORMS["inverse_normal"]["fn"] is t_inverse_normal
    assert TRANSFORMS["gumbel_cdf"]["fn"] is t_gumbel_cdf
    assert TRANSFORMS["frechet_cdf"]["fn"] is t_frechet_cdf
    assert TRANSFORMS["log_normal_cdf"]["fn"] is t_log_normal_cdf
    assert TRANSFORMS["return_period"]["fn"] is t_return_period
    assert TRANSFORMS["johnson_su"]["fn"] is t_johnson_su
    assert TRANSFORMS["gev_cdf"]["fn"] is t_gev_cdf
    assert TRANSFORMS["pareto_tail"]["fn"] is t_pareto_tail
    assert TRANSFORMS["log_logistic_cdf"]["fn"] is t_log_logistic_cdf
    assert TRANSFORMS["var_q95"]["fn"] is t_var_proxy
    assert TRANSFORMS["cvar_q95"]["fn"] is t_cvar
    assert TRANSFORMS["sharpe_proxy"]["fn"] is t_sharpe_proxy
    assert TRANSFORMS["drawdown"]["fn"] is t_drawdown
    assert TRANSFORMS["fold_change"]["fn"] is t_fold_change
    assert TRANSFORMS["excess_return"]["fn"] is t_excess_return
    assert TRANSFORMS["hellinger"]["fn"] is t_hellinger
    assert TRANSFORMS["chord_normalise"]["fn"] is t_chord_dist
    assert TRANSFORMS["relative_abundance"]["fn"] is t_relative_abundance
    assert TRANSFORMS["log_ratio"]["fn"] is t_log_ratio
    assert TRANSFORMS["weibull_reliability"]["fn"] is t_weibull_reliability
    assert TRANSFORMS["fatigue_miner"]["fn"] is t_fatigue_miner
    assert TRANSFORMS["rankine_failure"]["fn"] is t_rankine_failure
    assert TRANSFORMS["von_mises_stress"]["fn"] is t_von_mises
    assert TRANSFORMS["safety_factor"]["fn"] is t_safety_factor
    assert TRANSFORMS["cumulative_damage"]["fn"] is t_cumulative_damage
    assert TRANSFORMS["stress_life"]["fn"] is t_stress_life
    assert TRANSFORMS["sigmoid_dose"]["fn"] is t_sigmoid_dose
    assert TRANSFORMS["hill_response"]["fn"] is t_hill_response
    assert TRANSFORMS["log_auc"]["fn"] is t_log_auc
    assert TRANSFORMS["emax_model"]["fn"] is t_emax_model
    assert TRANSFORMS["log_shift"]["fn"] is t_log_shift
    assert TRANSFORMS["power_law_beta2"]["fn"] is t_power_law
    assert TRANSFORMS["power_law_beta05"]["fn"] is t_power_law
    assert TRANSFORMS["box_cox_sqrt"]["fn"] is t_box_cox
    assert TRANSFORMS["box_cox_log"]["fn"] is t_box_cox
    assert TRANSFORMS["clipped_excess_q90"]["fn"] is t_clipped_excess
    assert TRANSFORMS["exceedance_q75"]["fn"] is t_exceed_q75
    assert TRANSFORMS["exceedance_q90"]["fn"] is t_exceed_q90
    assert TRANSFORMS["exceedance_q95"]["fn"] is t_exceed_q95
    assert TRANSFORMS["exceedance_q99"]["fn"] is t_exceed_q99
    assert TRANSFORMS["log2_shift"]["fn"] is t_log2_shift
    assert TRANSFORMS["log10_shift"]["fn"] is t_log10_shift
    assert TRANSFORMS["log_log"]["fn"] is t_log_log
    assert TRANSFORMS["anomaly_pct"]["fn"] is t_anomaly_pct
    assert TRANSFORMS["bias_correction"]["fn"] is t_bias_correction
    assert TRANSFORMS["quantile_delta"]["fn"] is t_quantile_delta
    assert TRANSFORMS["growing_degree_days"]["fn"] is t_growing_degree_days
    assert TRANSFORMS["std_precip_idx"]["fn"] is t_standardised_precip_idx
    assert TRANSFORMS["nash_sutcliffe"]["fn"] is t_nash_sutcliffe
    assert TRANSFORMS["pot_log"]["fn"] is t_pot_log
    assert TRANSFORMS["log_flow"]["fn"] is t_log_flow
    assert TRANSFORMS["gradient_magnitude"]["fn"] is t_gradient_magnitude


def test_apply_transform_matches_split_module_functions() -> None:
    y = np.linspace(-2.0, 2.0, 12, dtype=float).reshape(3, 4)

    np.testing.assert_allclose(apply_transform(y, "affine_a2_b1"), t_affine(y, a=2.0, b=1.0))
    np.testing.assert_allclose(apply_transform(y, "tanh_a03"), t_tanh_pointwise(y, alpha=0.3))
    np.testing.assert_allclose(apply_transform(y, "square_pointwise"), t_square_pointwise(y))
    np.testing.assert_allclose(apply_transform(y, "exp_pointwise"), t_exp_pointwise(y, scale=0.1))
    np.testing.assert_allclose(apply_transform(y, "relu_pointwise"), t_relu_pointwise(y))
    np.testing.assert_allclose(apply_transform(y, "log1p_positive"), t_log1p_abs(y))
    np.testing.assert_allclose(apply_transform(y, "sqrt_abs"), t_sqrt_abs(y))
    np.testing.assert_allclose(apply_transform(y, "abs_pointwise"), t_abs_pointwise(y))
    np.testing.assert_allclose(apply_transform(y, "cube_pointwise"), t_cube_pointwise(y))
    np.testing.assert_allclose(apply_transform(y, "erf_pointwise"), t_erf_pointwise(y, scale=0.5))
    np.testing.assert_allclose(apply_transform(y, "sin_pointwise"), t_sin_pointwise(y, freq=0.5))
    np.testing.assert_allclose(apply_transform(y, "cos_pointwise"), t_cos_pointwise(y, freq=0.5))
    np.testing.assert_allclose(
        apply_transform(y, "step_pointwise"), t_step_pointwise(y, threshold=0.0)
    )
    np.testing.assert_allclose(apply_transform(y, "log_abs_pointwise"), t_log_abs(y, eps=1.0))
    np.testing.assert_allclose(apply_transform(y, "poly4"), t_poly4(y, scale=0.05))
    np.testing.assert_allclose(apply_transform(y, "poly5"), t_poly5(y, scale=0.05))
    np.testing.assert_allclose(apply_transform(y, "poly6"), t_poly6(y, scale=0.05))
    np.testing.assert_allclose(apply_transform(y, "legendre_p3"), t_legendre_p3(y, scale=0.3))
    np.testing.assert_allclose(apply_transform(y, "chebyshev_t4"), t_chebyshev_t4(y, scale=0.2))
    np.testing.assert_allclose(apply_transform(y, "hermite_he2"), t_hermite_he2(y, scale=0.3))
    np.testing.assert_allclose(apply_transform(y, "hermite_he3"), t_hermite_he3(y, scale=0.3))
    np.testing.assert_allclose(apply_transform(y, "neg_square"), t_neg_square(y))
    np.testing.assert_allclose(apply_transform(y, "smooth_bump"), t_smooth_bump(y, width=3.0))
    np.testing.assert_allclose(apply_transform(y, "rational_quadratic"), t_rational_quadratic(y))
    np.testing.assert_allclose(apply_transform(y, "inverse_pointwise"), t_inverse_abs(y, eps=1.0))
    np.testing.assert_allclose(apply_transform(y, "atan2pi"), t_atan2pi(y, scale=1.0))
    np.testing.assert_allclose(apply_transform(y, "exp_neg_sq"), t_exp_neg_sq(y, scale=0.3))
    np.testing.assert_allclose(apply_transform(y, "exp_pos_sq"), t_exp_pos_sq(y, scale=0.2))
    np.testing.assert_allclose(apply_transform(y, "inverse_sq"), t_inverse_sq(y, eps=1.0))
    np.testing.assert_allclose(apply_transform(y, "power_exp"), t_power_exp(y, scale=0.1))
    np.testing.assert_allclose(apply_transform(y, "sinc"), t_sinc(y, scale=0.5))
    np.testing.assert_allclose(apply_transform(y, "sin_squared"), t_sin_squared(y, freq=0.5))
    np.testing.assert_allclose(apply_transform(y, "cos_squared"), t_cos_squared(y, freq=0.5))
    np.testing.assert_allclose(
        apply_transform(y, "damped_sin"), t_damped_sin(y, freq=0.5, decay=0.1)
    )
    np.testing.assert_allclose(apply_transform(y, "sawtooth"), t_sawtooth(y, period=4.0))
    np.testing.assert_allclose(apply_transform(y, "square_wave"), t_square_wave(y, period=4.0))
    np.testing.assert_allclose(
        apply_transform(y, "double_sin"), t_double_sin(y, freq1=0.3, freq2=0.7)
    )
    np.testing.assert_allclose(
        apply_transform(y, "sin_cos_product"), t_sin_cos_product(y, freq=0.5)
    )
    np.testing.assert_allclose(
        apply_transform(y, "softplus_b01"), t_softplus_pointwise(y, beta=0.1)
    )
    np.testing.assert_allclose(apply_transform(y, "cosh_pointwise"), t_cosh_pointwise(y, scale=0.1))
    np.testing.assert_allclose(apply_transform(y, "cbrt_pointwise"), t_cbrt_pointwise(y))
    np.testing.assert_allclose(
        apply_transform(y, "logistic_pointwise"), t_logistic_pointwise(y, k=1.0)
    )
    np.testing.assert_allclose(
        apply_transform(y, "arctan_pointwise"), t_arctan_pointwise(y, scale=1.0)
    )
    np.testing.assert_allclose(apply_transform(y, "sinh_pointwise"), t_sinh_pointwise(y, scale=0.1))
    np.testing.assert_allclose(apply_transform(y, "gompertz_cdf"), t_gompertz(y, b=1.0, c=0.5))
    np.testing.assert_allclose(
        apply_transform(y, "algebraic_sigmoid"), t_algebraic_sigmoid(y, scale=0.5)
    )
    np.testing.assert_allclose(apply_transform(y, "swish"), t_swish(y, beta=1.0))
    np.testing.assert_allclose(apply_transform(y, "mish"), t_mish(y))
    np.testing.assert_allclose(apply_transform(y, "selu"), t_selu(y, alpha=1.6733, lam=1.0507))
    np.testing.assert_allclose(apply_transform(y, "softsign"), t_softsign(y, scale=1.0))
    np.testing.assert_allclose(apply_transform(y, "bent_identity"), t_bent_identity(y, scale=0.5))
    np.testing.assert_allclose(apply_transform(y, "hard_sigmoid"), t_hard_sigmoid(y, scale=0.5))
    np.testing.assert_allclose(apply_transform(y, "hard_tanh"), t_hard_tanh(y, scale=0.3))
    np.testing.assert_allclose(apply_transform(y, "soft_threshold"), t_soft_threshold(y, lam=1.0))
    np.testing.assert_allclose(apply_transform(y, "hard_threshold"), t_hard_threshold(y, lam=1.0))
    np.testing.assert_allclose(apply_transform(y, "ramp"), t_ramp(y, lo=-1.0, hi=1.0))
    np.testing.assert_allclose(
        apply_transform(y, "spike_gaussian"), t_spike(y, center=0.0, width=1.0)
    )
    np.testing.assert_allclose(
        apply_transform(y, "breakpoint"),
        t_breakpoint(y, bp=0.0, slope_lo=0.5, slope_hi=2.0),
    )
    np.testing.assert_allclose(apply_transform(y, "hockey_stick"), t_hockey_stick(y, bp=0.0))
    np.testing.assert_allclose(apply_transform(y, "deadzone"), t_deadzone(y, half_width=1.0))
    np.testing.assert_allclose(apply_transform(y, "temporal_cumsum"), t_temporal_cumsum(y))
    np.testing.assert_allclose(apply_transform(y, "temporal_peak"), t_temporal_peak(y))
    np.testing.assert_allclose(apply_transform(y, "temporal_rms"), t_temporal_rms(y))
    np.testing.assert_allclose(apply_transform(y, "temporal_range"), t_temporal_range(y))
    np.testing.assert_allclose(apply_transform(y, "temporal_autocorr"), t_temporal_autocorr(y))
    np.testing.assert_allclose(
        apply_transform(y, "temporal_quantile_q10"), t_temporal_quantile(y, q=0.10)
    )
    np.testing.assert_allclose(
        apply_transform(y, "temporal_quantile_q50"), t_temporal_quantile(y, q=0.50)
    )
    np.testing.assert_allclose(
        apply_transform(y, "temporal_quantile_q90"), t_temporal_quantile(y, q=0.90)
    )
    np.testing.assert_allclose(apply_transform(y, "sample_variance"), t_sample_variance(y))
    np.testing.assert_allclose(apply_transform(y, "negentropy_proxy"), t_negentropy_proxy(y))
    np.testing.assert_allclose(apply_transform(y, "wasserstein_proxy"), t_wasserstein_proxy(y))
    np.testing.assert_allclose(apply_transform(y, "energy_distance"), t_energy_distance_proxy(y))
    np.testing.assert_allclose(
        apply_transform(y, "renyi_entropy_a2"), t_entropy_renyi(y, alpha=2.0, bins=20)
    )
    np.testing.assert_allclose(apply_transform(y, "sample_skewness"), t_sample_skewness(y))
    np.testing.assert_allclose(apply_transform(y, "sample_kurtosis"), t_sample_kurtosis(y))
    np.testing.assert_allclose(apply_transform(y, "percentile_q10"), t_percentile_q10(y))
    np.testing.assert_allclose(apply_transform(y, "percentile_q90"), t_percentile_q90(y))
    np.testing.assert_allclose(apply_transform(y, "iqr"), t_interquartile_range(y))
    np.testing.assert_allclose(apply_transform(y, "rank_transform"), t_rank_transform(y))
    np.testing.assert_allclose(
        apply_transform(y, "standardised_anomaly"), t_standardised_anomaly(y)
    )
    np.testing.assert_allclose(apply_transform(y, "entropy_proxy"), t_entropy_proxy(y))
    np.testing.assert_allclose(apply_transform(y, "softmax_shift"), t_softmax_shift(y))
    np.testing.assert_allclose(apply_transform(y, "min_max_normalise"), t_min_max_normalise(y))
    np.testing.assert_allclose(apply_transform(y, "robust_scale"), t_robust_scale(y))
    np.testing.assert_allclose(apply_transform(y, "clamp_sigma"), t_clamp_sigma(y, n_sigma=2.0))
    np.testing.assert_allclose(apply_transform(y, "quantile_transform"), t_quantile_normalise(y))
    np.testing.assert_allclose(
        apply_transform(y, "winsorise_q10_q90"), t_winsorise(y, low=0.10, high=0.90)
    )
    np.testing.assert_allclose(apply_transform(y, "inverse_normal"), t_inverse_normal(y))
    np.testing.assert_allclose(apply_transform(y, "gumbel_cdf"), t_gumbel_cdf(y))
    np.testing.assert_allclose(apply_transform(y, "frechet_cdf"), t_frechet_cdf(y, shape=2.0))
    np.testing.assert_allclose(apply_transform(y, "log_normal_cdf"), t_log_normal_cdf(y, sigma=0.5))
    np.testing.assert_allclose(apply_transform(y, "return_period"), t_return_period(y))
    np.testing.assert_allclose(apply_transform(y, "johnson_su"), t_johnson_su(y))
    np.testing.assert_allclose(apply_transform(y, "gev_cdf"), t_gev_cdf(y, xi=0.3))
    np.testing.assert_allclose(apply_transform(y, "pareto_tail"), t_pareto_tail(y, alpha=1.5))
    np.testing.assert_allclose(
        apply_transform(y, "log_logistic_cdf"), t_log_logistic_cdf(y, beta=2.0)
    )
    np.testing.assert_allclose(apply_transform(y, "anscombe"), t_anscombe(y))
    np.testing.assert_allclose(apply_transform(y, "freeman_tukey"), t_freeman_tukey(y))
    np.testing.assert_allclose(apply_transform(y, "asinh_vst"), t_asinh_vs(y, scale=0.5))
    np.testing.assert_allclose(apply_transform(y, "modulus_lam05"), t_modulus(y, lam=0.5))
    np.testing.assert_allclose(apply_transform(y, "dual_power_lam03"), t_dual_power(y, lam=0.3))
    np.testing.assert_allclose(apply_transform(y, "var_q95"), t_var_proxy(y, q=0.95))
    np.testing.assert_allclose(apply_transform(y, "cvar_q95"), t_cvar(y, q=0.95))
    np.testing.assert_allclose(apply_transform(y, "sharpe_proxy"), t_sharpe_proxy(y, rf=0.0))
    np.testing.assert_allclose(apply_transform(y, "drawdown"), t_drawdown(y))
    np.testing.assert_allclose(apply_transform(y, "fold_change"), t_fold_change(y, eps=1.0))
    np.testing.assert_allclose(apply_transform(y, "excess_return"), t_excess_return(y))
    np.testing.assert_allclose(
        apply_transform(y, "weibull_reliability"), t_weibull_reliability(y, shape=2.0)
    )
    np.testing.assert_allclose(apply_transform(y, "fatigue_miner"), t_fatigue_miner(y, m=3.0))
    np.testing.assert_allclose(apply_transform(y, "rankine_failure"), t_rankine_failure(y))
    np.testing.assert_allclose(apply_transform(y, "von_mises_stress"), t_von_mises(y))
    np.testing.assert_allclose(
        apply_transform(y, "safety_factor"), t_safety_factor(y, capacity=1.0)
    )
    np.testing.assert_allclose(
        apply_transform(y, "cumulative_damage"), t_cumulative_damage(y, m=3.0)
    )
    np.testing.assert_allclose(apply_transform(y, "stress_life"), t_stress_life(y, C=1e6, m=3.0))
    np.testing.assert_allclose(
        apply_transform(y, "sigmoid_dose"), t_sigmoid_dose(y, EC50_q=0.5, n_hill=4.0)
    )
    np.testing.assert_allclose(
        apply_transform(y, "hill_response"), t_hill_response(y, n=2.0, EC50_q=0.5)
    )
    np.testing.assert_allclose(apply_transform(y, "log_auc"), t_log_auc(y, eps=1.0))
    np.testing.assert_allclose(
        apply_transform(y, "emax_model"), t_emax_model(y, Emax=1.0, ED50_q=0.5, n=1.0)
    )
    np.testing.assert_allclose(apply_transform(y, "anomaly_pct"), t_anomaly_pct(y, eps=1.0))
    np.testing.assert_allclose(apply_transform(y, "bias_correction"), t_bias_correction(y))
    np.testing.assert_allclose(apply_transform(y, "quantile_delta"), t_quantile_delta(y, q=0.90))
    np.testing.assert_allclose(
        apply_transform(y, "growing_degree_days"), t_growing_degree_days(y, base=10.0)
    )
    np.testing.assert_allclose(apply_transform(y, "std_precip_idx"), t_standardised_precip_idx(y))
    np.testing.assert_allclose(apply_transform(y, "nash_sutcliffe"), t_nash_sutcliffe(y))
    np.testing.assert_allclose(apply_transform(y, "pot_log"), t_pot_log(y, q=0.90, eps=1.0))
    np.testing.assert_allclose(apply_transform(y, "log_flow"), t_log_flow(y, eps=0.01))
    np.testing.assert_allclose(apply_transform(y, "hellinger"), t_hellinger(y))
    np.testing.assert_allclose(apply_transform(y, "chord_normalise"), t_chord_dist(y))
    np.testing.assert_allclose(apply_transform(y, "relative_abundance"), t_relative_abundance(y))
    np.testing.assert_allclose(apply_transform(y, "log_ratio"), t_log_ratio(y, eps=1.0))

    spatial_y = np.linspace(-2.0, 2.0, 36, dtype=float).reshape(3, 3, 4)
    np.testing.assert_allclose(
        apply_transform(spatial_y, "gradient_magnitude"), t_gradient_magnitude(spatial_y)
    )
    np.testing.assert_allclose(
        apply_transform(spatial_y, "regional_mean"), t_regional_mean(spatial_y)
    )
    np.testing.assert_allclose(apply_transform(spatial_y, "block_2x2"), t_block_2x2(spatial_y))
    np.testing.assert_allclose(apply_transform(spatial_y, "block_4x4"), t_block_4x4(spatial_y))
    np.testing.assert_allclose(apply_transform(spatial_y, "block_8x8"), t_block_8x8(spatial_y))
    np.testing.assert_allclose(
        apply_transform(spatial_y, "exceedance_area"), t_exceedance_area(spatial_y, quantile=0.75)
    )
    np.testing.assert_allclose(
        apply_transform(spatial_y, "matern_smooth"),
        t_matern_smooth(spatial_y, length_scale=0.15, nu=1.5),
    )
    np.testing.assert_allclose(
        apply_transform(spatial_y, "laplacian_roughness"), t_laplacian_roughness(spatial_y)
    )
    np.testing.assert_allclose(
        apply_transform(spatial_y, "contour_exceedance"),
        t_contour_exceedance(spatial_y, quantile=0.75),
    )
    np.testing.assert_allclose(
        apply_transform(spatial_y, "isoline_length"),
        t_isoline_length(spatial_y, quantile=0.75),
    )


def test_focused_transform_modules_exist() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    assert (package_root / "transforms" / "linear.py").exists()
    assert (package_root / "transforms" / "pointwise.py").exists()
    assert (package_root / "transforms" / "nonlinear.py").exists()
    assert (package_root / "transforms" / "samplewise.py").exists()
    assert (package_root / "transforms" / "aggregation.py").exists()
    assert (package_root / "transforms" / "field_ops.py").exists()
    assert (package_root / "transforms" / "statistical.py").exists()
    assert (package_root / "transforms" / "engineering.py").exists()
    assert (package_root / "transforms" / "environmental.py").exists()
    assert (package_root / "transforms" / "mathematical.py").exists()
    assert (package_root / "transforms" / "financial.py").exists()
    assert (package_root / "transforms" / "pharmacological.py").exists()
    assert (package_root / "transforms" / "ecological.py").exists()


def test_engineering_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_weibull_reliability(" not in monolith
    assert "def t_fatigue_miner(" not in monolith
    assert "def t_rankine_failure(" not in monolith
    assert "def t_von_mises(" not in monolith
    assert "def t_safety_factor(" not in monolith
    assert "def t_cumulative_damage(" not in monolith
    assert "def t_stress_life(" not in monolith


def test_elementary_pointwise_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_cube_pointwise(" not in monolith
    assert "def t_erf_pointwise(" not in monolith
    assert "def t_sin_pointwise(" not in monolith
    assert "def t_cos_pointwise(" not in monolith
    assert "def t_step_pointwise(" not in monolith
    assert "def t_log_abs(" not in monolith


def test_periodic_pointwise_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_sinc(" not in monolith
    assert "def t_sin_squared(" not in monolith
    assert "def t_cos_squared(" not in monolith
    assert "def t_damped_sin(" not in monolith
    assert "def t_sawtooth(" not in monolith
    assert "def t_square_wave(" not in monolith
    assert "def t_double_sin(" not in monolith
    assert "def t_sin_cos_product(" not in monolith


def test_nonlinear_pointwise_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_cosh_pointwise(" not in monolith
    assert "def t_cbrt_pointwise(" not in monolith
    assert "def t_logistic_pointwise(" not in monolith
    assert "def t_arctan_pointwise(" not in monolith
    assert "def t_sinh_pointwise(" not in monolith


def test_nonlinear_activation_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_gompertz(" not in monolith
    assert "def t_algebraic_sigmoid(" not in monolith
    assert "def t_swish(" not in monolith
    assert "def t_mish(" not in monolith
    assert "def t_selu(" not in monolith
    assert "def t_softsign(" not in monolith
    assert "def t_bent_identity(" not in monolith
    assert "def t_hard_sigmoid(" not in monolith
    assert "def t_hard_tanh(" not in monolith


def test_threshold_piecewise_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_soft_threshold(" not in monolith
    assert "def t_hard_threshold(" not in monolith
    assert "def t_ramp(" not in monolith
    assert "def t_spike(" not in monolith
    assert "def t_breakpoint(" not in monolith
    assert "def t_hockey_stick(" not in monolith
    assert "def t_deadzone(" not in monolith


def test_pharmacological_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_sigmoid_dose(" not in monolith
    assert "def t_hill_response(" not in monolith
    assert "def t_log_auc(" not in monolith
    assert "def t_emax_model(" not in monolith


def test_environmental_log_power_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_log_shift(" not in monolith
    assert "def t_power_law(" not in monolith
    assert "def t_box_cox(" not in monolith
    assert "def t_clipped_excess(" not in monolith
    assert "def _exceed(" not in monolith
    assert "def t_exceed_q75(" not in monolith
    assert "def t_exceed_q90(" not in monolith
    assert "def t_exceed_q95(" not in monolith
    assert "def t_exceed_q99(" not in monolith
    assert "def t_log2_shift(" not in monolith
    assert "def t_log10_shift(" not in monolith
    assert "def t_log_log(" not in monolith


def test_climate_bias_anomaly_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_anomaly_pct(" not in monolith
    assert "def t_bias_correction(" not in monolith
    assert "def t_quantile_delta(" not in monolith


def test_environmental_hydrology_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_growing_degree_days(" not in monolith
    assert "def t_standardised_precip_idx(" not in monolith
    assert "def t_nash_sutcliffe(" not in monolith
    assert "def t_pot_log(" not in monolith
    assert "def t_log_flow(" not in monolith


def test_mathematical_polynomial_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_poly4(" not in monolith
    assert "def t_poly5(" not in monolith
    assert "def t_poly6(" not in monolith
    assert "def t_legendre_p3(" not in monolith
    assert "def t_chebyshev_t4(" not in monolith
    assert "def t_hermite_he2(" not in monolith
    assert "def t_hermite_he3(" not in monolith


def test_mathematical_response_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_neg_square(" not in monolith
    assert "def t_smooth_bump(" not in monolith
    assert "def t_rational_quadratic(" not in monolith
    assert "def t_inverse_abs(" not in monolith
    assert "def t_atan2pi(" not in monolith
    assert "def t_exp_neg_sq(" not in monolith
    assert "def t_exp_pos_sq(" not in monolith
    assert "def t_inverse_sq(" not in monolith
    assert "def t_power_exp(" not in monolith


def test_variance_stabilising_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_anscombe(" not in monolith
    assert "def t_freeman_tukey(" not in monolith
    assert "def t_asinh_vs(" not in monolith
    assert "def t_modulus(" not in monolith
    assert "def t_dual_power(" not in monolith


def test_financial_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_var_proxy(" not in monolith
    assert "def t_cvar(" not in monolith
    assert "def t_sharpe_proxy(" not in monolith
    assert "def t_drawdown(" not in monolith
    assert "def t_fold_change(" not in monolith
    assert "def t_excess_return(" not in monolith


def test_ecological_family_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_hellinger(" not in monolith
    assert "def t_chord_dist(" not in monolith
    assert "def t_relative_abundance(" not in monolith
    assert "def t_log_ratio(" not in monolith


def test_spatial_aggregation_and_field_ops_no_longer_defined_in_monolith() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    monolith = (package_root / "transforms" / "transforms.py").read_text(encoding="utf-8")

    assert "def t_regional_mean(" not in monolith
    assert "def _block_avg(" not in monolith
    assert "def t_block_2x2(" not in monolith
    assert "def t_block_4x4(" not in monolith
    assert "def t_block_8x8(" not in monolith
    assert "def t_exceedance_area(" not in monolith
    assert "def t_matern_smooth(" not in monolith
    assert "def t_laplacian_roughness(" not in monolith
    assert "def t_contour_exceedance(" not in monolith
    assert "def t_isoline_length(" not in monolith
