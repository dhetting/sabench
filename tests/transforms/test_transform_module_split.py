from __future__ import annotations

from pathlib import Path

import numpy as np

import sabench
from sabench.transforms import TRANSFORMS, apply_transform, get_transform_spec
from sabench.transforms.aggregation import (
    t_energy_distance_proxy,
    t_entropy_renyi,
    t_interquartile_range,
    t_negentropy_proxy,
    t_percentile_q10,
    t_percentile_q90,
    t_sample_kurtosis,
    t_sample_skewness,
    t_sample_variance,
    t_temporal_peak,
    t_wasserstein_proxy,
)
from sabench.transforms.field_ops import t_gradient_magnitude
from sabench.transforms.linear import t_affine
from sabench.transforms.nonlinear import t_softplus_pointwise
from sabench.transforms.pointwise import (
    t_abs_pointwise,
    t_exp_pointwise,
    t_log1p_abs,
    t_relu_pointwise,
    t_sqrt_abs,
    t_square_pointwise,
    t_tanh_pointwise,
)
from sabench.transforms.samplewise import t_temporal_cumsum


def test_representative_transform_specs_point_to_split_modules() -> None:
    assert get_transform_spec("affine_a2_b1").module == "sabench.transforms.linear"
    assert get_transform_spec("tanh_a03").module == "sabench.transforms.pointwise"
    assert get_transform_spec("square_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("exp_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("relu_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("log1p_positive").module == "sabench.transforms.pointwise"
    assert get_transform_spec("sqrt_abs").module == "sabench.transforms.pointwise"
    assert get_transform_spec("abs_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("softplus_b01").module == "sabench.transforms.nonlinear"
    assert get_transform_spec("temporal_cumsum").module == "sabench.transforms.samplewise"
    assert get_transform_spec("temporal_peak").module == "sabench.transforms.aggregation"
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
    assert get_transform_spec("gradient_magnitude").module == "sabench.transforms.field_ops"


def test_legacy_transform_registry_uses_split_module_functions() -> None:
    assert TRANSFORMS["affine_a2_b1"]["fn"] is t_affine
    assert TRANSFORMS["tanh_a03"]["fn"] is t_tanh_pointwise
    assert TRANSFORMS["square_pointwise"]["fn"] is t_square_pointwise
    assert TRANSFORMS["exp_pointwise"]["fn"] is t_exp_pointwise
    assert TRANSFORMS["relu_pointwise"]["fn"] is t_relu_pointwise
    assert TRANSFORMS["log1p_positive"]["fn"] is t_log1p_abs
    assert TRANSFORMS["sqrt_abs"]["fn"] is t_sqrt_abs
    assert TRANSFORMS["abs_pointwise"]["fn"] is t_abs_pointwise
    assert TRANSFORMS["softplus_b01"]["fn"] is t_softplus_pointwise
    assert TRANSFORMS["temporal_cumsum"]["fn"] is t_temporal_cumsum
    assert TRANSFORMS["temporal_peak"]["fn"] is t_temporal_peak
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
    np.testing.assert_allclose(apply_transform(y, "softplus_b01"), t_softplus_pointwise(y, beta=0.1))
    np.testing.assert_allclose(apply_transform(y, "temporal_cumsum"), t_temporal_cumsum(y))
    np.testing.assert_allclose(apply_transform(y, "temporal_peak"), t_temporal_peak(y))
    np.testing.assert_allclose(apply_transform(y, "sample_variance"), t_sample_variance(y))
    np.testing.assert_allclose(apply_transform(y, "negentropy_proxy"), t_negentropy_proxy(y))
    np.testing.assert_allclose(apply_transform(y, "wasserstein_proxy"), t_wasserstein_proxy(y))
    np.testing.assert_allclose(apply_transform(y, "energy_distance"), t_energy_distance_proxy(y))
    np.testing.assert_allclose(apply_transform(y, "renyi_entropy_a2"), t_entropy_renyi(y, alpha=2.0, bins=20))
    np.testing.assert_allclose(apply_transform(y, "sample_skewness"), t_sample_skewness(y))
    np.testing.assert_allclose(apply_transform(y, "sample_kurtosis"), t_sample_kurtosis(y))
    np.testing.assert_allclose(apply_transform(y, "percentile_q10"), t_percentile_q10(y))
    np.testing.assert_allclose(apply_transform(y, "percentile_q90"), t_percentile_q90(y))
    np.testing.assert_allclose(apply_transform(y, "iqr"), t_interquartile_range(y))

    spatial_y = np.linspace(-2.0, 2.0, 36, dtype=float).reshape(3, 3, 4)
    np.testing.assert_allclose(apply_transform(spatial_y, "gradient_magnitude"), t_gradient_magnitude(spatial_y))


def test_focused_transform_modules_exist() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    assert (package_root / "transforms" / "linear.py").exists()
    assert (package_root / "transforms" / "pointwise.py").exists()
    assert (package_root / "transforms" / "nonlinear.py").exists()
    assert (package_root / "transforms" / "samplewise.py").exists()
    assert (package_root / "transforms" / "aggregation.py").exists()
    assert (package_root / "transforms" / "field_ops.py").exists()
