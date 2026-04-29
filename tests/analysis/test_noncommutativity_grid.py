"""Tests for registry-driven noncommutativity grid utilities."""

from __future__ import annotations

import numpy as np
import pytest

from sabench.analysis.grid import (
    as_estimator_output,
    classify_noncommutativity_pair,
    evaluate_noncommutativity_grid,
    evaluate_noncommutativity_pair,
    variance_weighted_sobol_profile,
)
from sabench.benchmarks import BENCHMARK_REGISTRY


def test_classify_noncommutativity_pair_includes_supported_output_kind() -> None:
    compatibility = classify_noncommutativity_pair("LinearModel", "affine_a2_b1")

    assert compatibility.pair_status == "included"
    assert compatibility.reason == "compatible output kind"
    assert compatibility.benchmark_output_kind == "scalar"
    assert compatibility.transform_mechanism == "pointwise"
    assert "affine" in compatibility.transform_tags
    assert compatibility.as_dict()["transform_tags"]


def test_classify_noncommutativity_pair_excludes_unsupported_output_kind() -> None:
    compatibility = classify_noncommutativity_pair("LinearModel", "regional_mean")

    assert compatibility.pair_status == "excluded"
    assert compatibility.benchmark_output_kind == "scalar"
    assert compatibility.transform_supported_output_kinds == ("spatial",)
    assert "not supported" in compatibility.reason


def test_as_estimator_output_flattens_multi_output_after_sample_axis() -> None:
    output = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    flattened = as_estimator_output(output)

    assert flattened.shape == (2, 12)
    np.testing.assert_array_equal(flattened[0], output[0].ravel())


def test_variance_weighted_sobol_profile_uses_output_variance_weights() -> None:
    indices = np.array([[0.2, 0.8], [0.6, 0.4]])
    output = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])

    profile = variance_weighted_sobol_profile(indices, output)

    np.testing.assert_allclose(profile, np.array([0.2, 0.6]))


def test_variance_weighted_sobol_profile_falls_back_to_mean_for_constant_outputs() -> None:
    indices = np.array([[0.2, 0.8], [0.6, 0.4]])
    output = np.ones((3, 2))

    profile = variance_weighted_sobol_profile(indices, output)

    np.testing.assert_allclose(profile, np.array([0.5, 0.5]))


def test_evaluate_noncommutativity_pair_computes_affine_reference_metrics() -> None:
    result = evaluate_noncommutativity_pair(
        "LinearModel",
        "affine_a2_b1",
        n_base=128,
        seed=123,
        tau=0.05,
        top_k=2,
    )
    row = result.as_dict()

    assert result.pair_status == "included"
    assert result.metrics_status == "computed"
    assert row["raw_output_finite"] is True
    assert row["transformed_output_finite"] is True
    assert row["decision_score_s1"] == pytest.approx(0.0, abs=1e-12)
    assert row["sensitivity_shift_s1"] == pytest.approx(0.0, abs=1e-12)
    assert row["decision_score_st"] == pytest.approx(0.0, abs=1e-12)
    assert row["sensitivity_shift_st"] == pytest.approx(0.0, abs=1e-12)
    assert row["topk_changed_s1"] is False
    assert row["topk_changed_st"] is False


def test_evaluate_noncommutativity_pair_returns_row_for_excluded_pair() -> None:
    result = evaluate_noncommutativity_pair(
        "LinearModel",
        "regional_mean",
        n_base=16,
        seed=123,
    )
    row = result.as_dict()

    assert result.pair_status == "excluded"
    assert result.metrics_status == "not_applicable"
    assert "not supported" in result.reason
    assert row["n_inputs"] == BENCHMARK_REGISTRY["LinearModel"].spec.d
    assert row["n_evaluations"] is None
    assert "decision_score_s1" not in row


def test_evaluate_noncommutativity_grid_returns_deterministic_rows() -> None:
    results = evaluate_noncommutativity_grid(
        benchmark_keys=("LinearModel",),
        transform_keys=("affine_a2_b1", "regional_mean"),
        n_base=32,
        seed=123,
    )

    assert [result.transform_key for result in results] == ["affine_a2_b1", "regional_mean"]
    assert [result.metrics_status for result in results] == ["computed", "not_applicable"]
    assert all(result.benchmark_key == "LinearModel" for result in results)


@pytest.mark.parametrize("bad_n_base", [0, -1])
def test_evaluate_noncommutativity_pair_rejects_nonpositive_n_base(bad_n_base: int) -> None:
    with pytest.raises(ValueError, match="n_base"):
        evaluate_noncommutativity_pair("LinearModel", "affine_a2_b1", n_base=bad_n_base)
