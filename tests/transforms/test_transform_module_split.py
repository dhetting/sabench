from __future__ import annotations

from pathlib import Path

import numpy as np

import sabench
from sabench.transforms import TRANSFORMS, apply_transform, get_transform_spec
from sabench.transforms.aggregation import t_temporal_peak
from sabench.transforms.field_ops import t_gradient_magnitude
from sabench.transforms.linear import t_affine
from sabench.transforms.nonlinear import (
    t_algebraic_sigmoid,
    t_arctan_pointwise,
    t_bent_identity,
    t_cbrt_pointwise,
    t_cosh_pointwise,
    t_gompertz,
    t_hard_sigmoid,
    t_hard_tanh,
    t_logistic_pointwise,
    t_mish,
    t_selu,
    t_sinh_pointwise,
    t_softplus_pointwise,
    t_softsign,
    t_swish,
)
from sabench.transforms.pointwise import (
    t_abs_pointwise,
    t_cos_pointwise,
    t_cube_pointwise,
    t_erf_pointwise,
    t_exp_pointwise,
    t_log1p_abs,
    t_log_abs,
    t_relu_pointwise,
    t_sin_pointwise,
    t_sqrt_abs,
    t_square_pointwise,
    t_step_pointwise,
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
    assert get_transform_spec("cube_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("erf_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("sin_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("cos_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("step_pointwise").module == "sabench.transforms.pointwise"
    assert get_transform_spec("log_abs_pointwise").module == "sabench.transforms.pointwise"
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
    assert get_transform_spec("temporal_cumsum").module == "sabench.transforms.samplewise"
    assert get_transform_spec("temporal_peak").module == "sabench.transforms.aggregation"
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
    assert TRANSFORMS["cube_pointwise"]["fn"] is t_cube_pointwise
    assert TRANSFORMS["erf_pointwise"]["fn"] is t_erf_pointwise
    assert TRANSFORMS["sin_pointwise"]["fn"] is t_sin_pointwise
    assert TRANSFORMS["cos_pointwise"]["fn"] is t_cos_pointwise
    assert TRANSFORMS["step_pointwise"]["fn"] is t_step_pointwise
    assert TRANSFORMS["log_abs_pointwise"]["fn"] is t_log_abs
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
    assert TRANSFORMS["temporal_cumsum"]["fn"] is t_temporal_cumsum
    assert TRANSFORMS["temporal_peak"]["fn"] is t_temporal_peak
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
    np.testing.assert_allclose(apply_transform(y, "temporal_cumsum"), t_temporal_cumsum(y))
    np.testing.assert_allclose(apply_transform(y, "temporal_peak"), t_temporal_peak(y))

    spatial_y = np.linspace(-2.0, 2.0, 36, dtype=float).reshape(3, 3, 4)
    np.testing.assert_allclose(
        apply_transform(spatial_y, "gradient_magnitude"), t_gradient_magnitude(spatial_y)
    )


def test_focused_transform_modules_exist() -> None:
    package_root = Path(sabench.__file__).resolve().parent
    assert (package_root / "transforms" / "linear.py").exists()
    assert (package_root / "transforms" / "pointwise.py").exists()
    assert (package_root / "transforms" / "nonlinear.py").exists()
    assert (package_root / "transforms" / "samplewise.py").exists()
    assert (package_root / "transforms" / "aggregation.py").exists()
    assert (package_root / "transforms" / "field_ops.py").exists()
