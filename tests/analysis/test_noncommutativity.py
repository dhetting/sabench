"""Tests for empirical Sobol-profile noncommutativity metrics."""

from __future__ import annotations

import numpy as np
import pytest

from sabench.analysis import (
    ProfileShiftMetrics,
    decision_score,
    profile_shift_summary,
    sensitivity_shift,
    sobol_profile_shift_metrics,
    soft_threshold,
    spearman_rank_correlation,
    threshold_flip_count,
    topk_changed,
)


def test_soft_threshold_matches_paper_sigmoid() -> None:
    values = np.array([0.0, 0.05, 0.10])
    expected = 1.0 / (1.0 + np.exp(-((values - 0.05) / 0.05)))

    actual = soft_threshold(values, tau=0.05)

    np.testing.assert_allclose(actual, expected)
    assert actual[0] < actual[1] < actual[2]
    assert actual[1] == pytest.approx(0.5)


@pytest.mark.parametrize("bad_tau", [0.0, -0.1, np.inf, np.nan])
def test_soft_threshold_rejects_invalid_tau(bad_tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        soft_threshold(np.array([0.1]), tau=bad_tau)


def test_decision_score_is_zero_for_identical_profiles() -> None:
    profile = np.array([0.01, 0.05, 0.20])

    assert decision_score(profile, profile, tau=0.05) == pytest.approx(0.0)


def test_decision_score_matches_mean_soft_classification_shift() -> None:
    source = np.array([0.01, 0.05, 0.20])
    transformed = np.array([0.10, 0.02, 0.20])
    source_soft = 1.0 / (1.0 + np.exp(-((source - 0.05) / 0.05)))
    transformed_soft = 1.0 / (1.0 + np.exp(-((transformed - 0.05) / 0.05)))
    expected = np.mean(np.abs(transformed_soft - source_soft))

    assert decision_score(source, transformed, tau=0.05) == pytest.approx(expected)


def test_sensitivity_shift_matches_bray_curtis_formula() -> None:
    source = np.array([0.2, 0.3, 0.5])
    transformed = np.array([0.1, 0.6, 0.3])
    expected = np.abs(transformed - source).sum() / (source.sum() + transformed.sum())

    actual = sensitivity_shift(source, transformed)

    assert actual == pytest.approx(expected)
    assert 0.0 <= actual <= 1.0


def test_sensitivity_shift_rejects_zero_total_mass() -> None:
    with pytest.raises(ValueError, match="positive total sensitivity mass"):
        sensitivity_shift(np.array([0.0, 0.0]), np.array([0.0, 0.0]))


@pytest.mark.parametrize(
    ("source", "transformed", "error"),
    [
        (np.array([[0.1, 0.2]]), np.array([0.1, 0.2]), "one-dimensional"),
        (np.array([0.1]), np.array([0.1, 0.2]), "same shape"),
        (np.array([0.1, np.nan]), np.array([0.1, 0.2]), "finite"),
        (np.array([0.1, -0.01]), np.array([0.1, 0.2]), "nonnegative"),
        (np.array([]), np.array([]), "must not be empty"),
    ],
)
def test_profile_metrics_validate_profile_inputs(
    source: np.ndarray,
    transformed: np.ndarray,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        decision_score(source, transformed)


def test_threshold_flip_count_uses_hard_threshold_crossings() -> None:
    source = np.array([0.01, 0.04, 0.06, 0.20])
    transformed = np.array([0.10, 0.03, 0.04, 0.25])

    assert threshold_flip_count(source, transformed, tau=0.05) == 2


def test_topk_changed_compares_unordered_top_driver_sets() -> None:
    source = np.array([0.5, 0.3, 0.2, 0.0])
    same_top_two = np.array([0.4, 0.35, 0.1, 0.0])
    different_top_two = np.array([0.4, 0.1, 0.35, 0.0])

    assert not topk_changed(source, same_top_two, k=2)
    assert topk_changed(source, different_top_two, k=2)


@pytest.mark.parametrize("bad_k", [0, -1])
def test_topk_changed_rejects_nonpositive_k(bad_k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        topk_changed(np.array([0.1, 0.2]), np.array([0.2, 0.1]), k=bad_k)


def test_spearman_rank_correlation_handles_ordering_and_constants() -> None:
    increasing = np.array([0.1, 0.2, 0.3])
    decreasing = np.array([0.3, 0.2, 0.1])
    constant = np.array([0.2, 0.2, 0.2])

    assert spearman_rank_correlation(increasing, increasing) == pytest.approx(1.0)
    assert spearman_rank_correlation(increasing, decreasing) == pytest.approx(-1.0)
    assert spearman_rank_correlation(increasing, constant) is None


def test_profile_shift_summary_returns_standard_metric_bundle() -> None:
    source = np.array([0.2, 0.3, 0.5])
    transformed = np.array([0.6, 0.3, 0.1])

    summary = profile_shift_summary(source, transformed, tau=0.05, top_k=1)

    assert isinstance(summary, ProfileShiftMetrics)
    assert summary.decision_score == pytest.approx(decision_score(source, transformed))
    assert summary.sensitivity_shift == pytest.approx(sensitivity_shift(source, transformed))
    assert summary.threshold_flip_count == 0
    assert summary.topk_changed
    assert summary.max_abs_shift == pytest.approx(0.4)
    assert summary.mean_abs_shift == pytest.approx(np.array([0.4, 0.0, 0.4]).mean())
    assert summary.top_source_index == 2
    assert summary.top_transformed_index == 0
    assert summary.as_dict()["decision_score"] == pytest.approx(summary.decision_score)


def test_sobol_profile_shift_metrics_flattens_s1_and_st_results() -> None:
    source_s1 = np.array([0.2, 0.3, 0.5])
    transformed_s1 = np.array([0.6, 0.3, 0.1])
    source_st = np.array([0.4, 0.4, 0.6])
    transformed_st = np.array([0.4, 0.5, 0.5])

    metrics = sobol_profile_shift_metrics(
        source_s1,
        transformed_s1,
        source_st,
        transformed_st,
        tau=0.05,
        top_k=2,
    )

    assert metrics["decision_score_s1"] == pytest.approx(decision_score(source_s1, transformed_s1))
    assert metrics["sensitivity_shift_st"] == pytest.approx(
        sensitivity_shift(source_st, transformed_st)
    )
    assert metrics["top_source_index_s1"] == 2
    assert metrics["top_transformed_index_st"] == 1
    assert {key.rsplit("_", maxsplit=1)[-1] for key in metrics} == {"s1", "st"}
