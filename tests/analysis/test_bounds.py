"""Tests for Taylor-reference Sobol perturbation-bound utilities."""

from __future__ import annotations

import numpy as np
import pytest

from sabench.analysis.bounds import (
    classify_bounds_applicability,
    get_smooth_pointwise_analysis,
    local_affine_diagnostics,
    local_affine_perturbation_bound,
    projection_perturbation_bound,
    sufficient_taylor_eta,
    supported_smooth_pointwise_transform_keys,
    taylor_reference_diagnostics,
    taylor_reference_values,
    taylor_residual_values,
)


def test_projection_perturbation_bound_matches_theorem_formula() -> None:
    eta = 0.1
    p = np.array([0.0, 0.25, 1.0])
    expected = (2 * eta * np.sqrt(p) * (1 + np.sqrt(p)) + eta**2 * (1 + p)) / ((1 - eta) ** 2)

    actual = projection_perturbation_bound(eta, p, cap=False)

    np.testing.assert_allclose(actual, expected)


def test_projection_perturbation_bound_can_cap_at_one() -> None:
    assert projection_perturbation_bound(0.9, 1.0, cap=True) == pytest.approx(1.0)


@pytest.mark.parametrize("bad_eta", [-0.1, 1.0, np.inf, np.nan])
def test_projection_perturbation_bound_rejects_invalid_eta(bad_eta: float) -> None:
    with pytest.raises(ValueError, match="eta"):
        projection_perturbation_bound(bad_eta, 0.5)


@pytest.mark.parametrize("bad_p", [-0.1, 1.1, np.inf, np.nan])
def test_projection_perturbation_bound_rejects_invalid_p(bad_p: float) -> None:
    with pytest.raises(ValueError, match="p"):
        projection_perturbation_bound(0.2, bad_p)


def test_local_affine_bound_matches_corollary_formula() -> None:
    lambda_value = 0.2
    p = 0.25
    expected = (lambda_value * np.sqrt(p) * (1 + np.sqrt(p)) + 0.25 * lambda_value**2 * (1 + p)) / (
        1 - lambda_value / 2
    ) ** 2

    actual = local_affine_perturbation_bound(lambda_value, p, cap=False)

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("bad_lambda", [-0.1, 2.0, np.inf, np.nan])
def test_local_affine_bound_rejects_invalid_lambda(bad_lambda: float) -> None:
    with pytest.raises(ValueError, match="lambda_value"):
        local_affine_perturbation_bound(bad_lambda, 0.5)


def test_supported_derivative_registry_includes_initial_theorem_transforms() -> None:
    keys = set(supported_smooth_pointwise_transform_keys())

    assert {"affine_a2_b1", "square_pointwise", "cube_pointwise"} <= keys
    assert "abs_pointwise" not in keys


def test_taylor_reference_for_square_is_exact_at_second_order() -> None:
    y = np.array([-1.0, 0.0, 2.0, 3.0])
    analysis = get_smooth_pointwise_analysis("square_pointwise")

    reference = taylor_reference_values(y, analysis, order=2)
    residual = taylor_residual_values(y, analysis, order=2)
    diagnostics = taylor_reference_diagnostics(y, analysis, order=2)

    expected_reference = y**2 - np.mean(y) ** 2
    np.testing.assert_allclose(reference, expected_reference)
    np.testing.assert_allclose(residual, np.zeros_like(y))
    assert diagnostics.status == "computed"
    assert diagnostics.eta_empirical == pytest.approx(0.0)
    assert diagnostics.eta_sufficient == pytest.approx(0.0)
    assert diagnostics.eta_empirical_lt_one
    assert diagnostics.eta_sufficient_lt_one


def test_taylor_reference_for_affine_transform_has_zero_residual() -> None:
    y = np.array([-2.0, -1.0, 1.0, 3.0])
    analysis = get_smooth_pointwise_analysis("affine_a2_b1")

    diagnostics = taylor_reference_diagnostics(y, analysis, order=1)

    assert diagnostics.status == "computed"
    assert diagnostics.sigma_reference > 0.0
    assert diagnostics.eta_empirical == pytest.approx(0.0)
    assert diagnostics.as_summary_dict()["bound_status"] == "computed"


def test_taylor_reference_diagnostics_flags_zero_reference_variance() -> None:
    y = np.array([-1.0, 1.0, -1.0, 1.0])
    analysis = get_smooth_pointwise_analysis("square_pointwise")

    diagnostics = taylor_reference_diagnostics(y, analysis, order=2)

    assert diagnostics.status == "reference_zero_variance"
    assert diagnostics.eta_empirical is None
    assert diagnostics.eta_sufficient is None


def test_sufficient_eta_uses_supremum_derivative_and_sample_moment() -> None:
    y = np.array([-1.0, 0.0, 2.0, 3.0])
    analysis = get_smooth_pointwise_analysis("cube_pointwise")
    reference = taylor_reference_values(y, analysis, order=2)
    sigma_reference = float(np.sqrt(np.mean((reference - reference.mean()) ** 2)))
    centered = y - y.mean()
    expected = 6.0 * np.sqrt(np.mean(np.abs(centered) ** 6)) / (6.0 * sigma_reference)

    actual = sufficient_taylor_eta(
        y,
        analysis,
        order=2,
        sigma_reference=sigma_reference,
        support=(-1.0, 3.0),
    )

    assert actual == pytest.approx(expected)


def test_local_affine_diagnostics_for_affine_transform_has_zero_lambda() -> None:
    y = np.array([-2.0, -1.0, 1.0, 3.0])
    analysis = get_smooth_pointwise_analysis("affine_a2_b1")

    diagnostics = local_affine_diagnostics(y, analysis, support=(-2.0, 3.0))

    assert diagnostics.status == "computed"
    assert diagnostics.support_source == "provided_support"
    assert diagnostics.phi_prime_mu == pytest.approx(2.0)
    assert diagnostics.rho2 == pytest.approx(0.0)
    assert diagnostics.lambda_value == pytest.approx(0.0)
    assert diagnostics.eta_upper == pytest.approx(0.0)
    assert diagnostics.lambda_lt_two


def test_local_affine_diagnostics_flags_zero_slope_for_critical_point() -> None:
    y = np.array([-1.0, 0.0, 1.0])
    analysis = get_smooth_pointwise_analysis("square_pointwise")

    diagnostics = local_affine_diagnostics(y, analysis)

    assert diagnostics.status == "zero_slope"
    assert diagnostics.lambda_value is None


def test_classify_bounds_applicability_separates_theorem_assumptions() -> None:
    supported = classify_bounds_applicability(
        output_kind="scalar",
        mechanism="pointwise",
        tags=("smooth", "pointwise"),
        transform_key="square_pointwise",
    )
    non_scalar = classify_bounds_applicability(
        output_kind="spatial",
        mechanism="pointwise",
        tags=("smooth", "pointwise"),
        transform_key="square_pointwise",
    )
    non_pointwise = classify_bounds_applicability(
        output_kind="scalar",
        mechanism="samplewise",
        tags=("smooth",),
        transform_key="rank_transform",
    )
    non_smooth = classify_bounds_applicability(
        output_kind="scalar",
        mechanism="pointwise",
        tags=("nonsmooth", "pointwise"),
        transform_key="abs_pointwise",
    )
    missing_metadata = classify_bounds_applicability(
        output_kind="scalar",
        mechanism="pointwise",
        tags=("smooth", "pointwise"),
        transform_key="tanh_a03",
    )

    assert supported.supported
    assert non_scalar.status == "bounds_not_scalar_output"
    assert non_pointwise.status == "bounds_not_pointwise"
    assert non_smooth.status == "bounds_not_smooth"
    assert missing_metadata.status == "bounds_no_derivative_metadata"


@pytest.mark.parametrize(
    "bad_y",
    [np.array([[1.0, 2.0]]), np.array([]), np.array([1.0, np.nan])],
)
def test_taylor_diagnostics_reject_invalid_output_samples(bad_y: np.ndarray) -> None:
    analysis = get_smooth_pointwise_analysis("square_pointwise")

    with pytest.raises(ValueError):
        taylor_reference_diagnostics(bad_y, analysis, order=1)
