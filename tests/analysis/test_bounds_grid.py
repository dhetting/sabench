"""Tests for registry-driven bounds-theorem grid utilities."""

from __future__ import annotations

import pytest

from sabench.analysis.bounds_grid import (
    BENCHMARK_OUTPUT_BOUNDS,
    BOUNDS_STATUSES,
    classify_bounds_grid_pair,
    evaluate_bounds_grid,
    evaluate_bounds_pair,
)


def test_bounds_status_catalog_includes_notebook_contract_statuses() -> None:
    assert set(BOUNDS_STATUSES) == {
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
    }


def test_classify_bounds_grid_pair_separates_static_theorem_assumptions() -> None:
    supported = classify_bounds_grid_pair("LinearModel", "affine_a2_b1")
    non_scalar = classify_bounds_grid_pair("Campbell2D", "square_pointwise")
    non_pointwise = classify_bounds_grid_pair("LinearModel", "rank_transform")
    non_smooth = classify_bounds_grid_pair("LinearModel", "abs_pointwise")
    # tanh_a03 previously had no derivative metadata; it now has full metadata.
    tanh_supported = classify_bounds_grid_pair("LinearModel", "tanh_a03")

    assert supported.static_status == "bounds_supported"
    assert non_scalar.static_status == "bounds_not_scalar_output"
    assert non_pointwise.static_status == "bounds_not_pointwise"
    assert non_smooth.static_status == "bounds_not_smooth"
    assert tanh_supported.static_status == "bounds_supported"


def test_evaluate_bounds_pair_marks_sample_range_results_as_diagnostics() -> None:
    result = evaluate_bounds_pair(
        "LinearModel",
        "affine_a2_b1",
        n_base=64,
        seed=123,
        taylor_order=1,
    )
    row = result.as_dict()

    assert result.bounds_status == "bounds_diagnostic_sample_support"
    assert row["support_source"] == "sample_range"
    assert row["taylor_status"] == "computed"
    assert row["eta_empirical"] == pytest.approx(0.0)
    assert row["projection_bound_s1_max"] == pytest.approx(0.0)
    assert row["projection_bound_st_max"] == pytest.approx(0.0)
    assert row["local_affine_status"] == "computed"
    assert row["lambda_value"] == pytest.approx(0.0)


def test_evaluate_bounds_pair_marks_provided_support_as_theorem_supported() -> None:
    result = evaluate_bounds_pair(
        "LinearModel",
        "affine_a2_b1",
        n_base=64,
        seed=123,
        taylor_order=1,
        support=(-100.0, 100.0),
    )

    assert result.bounds_status == "bounds_supported"
    assert result.as_dict()["support_source"] == "provided_support"


def test_evaluate_bounds_pair_returns_non_applicable_static_status() -> None:
    result = evaluate_bounds_pair("Campbell2D", "square_pointwise", n_base=16, seed=123)
    row = result.as_dict()

    assert result.bounds_status == "bounds_not_scalar_output"
    assert row["n_evaluations"] is None
    assert "eta_empirical" not in row


def test_evaluate_bounds_pair_reports_invalid_support_as_domain_invalid() -> None:
    result = evaluate_bounds_pair(
        "LinearModel",
        "affine_a2_b1",
        n_base=32,
        seed=123,
        taylor_order=1,
        support=(100.0, 101.0),
    )

    assert result.bounds_status == "bounds_domain_invalid"
    assert "support must contain" in result.reason


def test_evaluate_bounds_grid_returns_deterministic_rows() -> None:
    results = evaluate_bounds_grid(
        benchmark_keys=("LinearModel",),
        transform_keys=("affine_a2_b1", "abs_pointwise"),
        n_base=32,
        seed=123,
        taylor_order=1,
    )

    assert [result.transform_key for result in results] == ["affine_a2_b1", "abs_pointwise"]
    assert [result.bounds_status for result in results] == [
        "bounds_diagnostic_sample_support",
        "bounds_not_smooth",
    ]


def test_benchmark_output_bounds_covers_all_scalar_benchmarks_with_formulas() -> None:
    """BENCHMARK_OUTPUT_BOUNDS entries are non-empty and each has lower < upper."""
    assert len(BENCHMARK_OUTPUT_BOUNDS) >= 5
    for key, (lo, hi) in BENCHMARK_OUTPUT_BOUNDS.items():
        assert lo < hi, f"{key}: lower={lo} must be strictly less than upper={hi}"


def test_benchmark_output_bounds_contain_sampled_values() -> None:
    """Each analytical bound must contain all values from a large sample."""
    import numpy as np

    from sabench.benchmarks.registry import BENCHMARK_REGISTRY

    rng = np.random.default_rng(42)
    N = 4096
    for key, (lo, hi) in BENCHMARK_OUTPUT_BOUNDS.items():
        bench = BENCHMARK_REGISTRY[key].benchmark_cls()
        seed = int(rng.integers(0, 2**31))
        X = bench.sample(N, seed=seed)
        Y = bench.evaluate(X).flatten()
        y_min = float(np.min(Y))
        y_max = float(np.max(Y))
        assert y_min >= lo, f"{key}: observed min {y_min:.6f} < theoretical lower {lo:.6f}"
        assert y_max <= hi, f"{key}: observed max {y_max:.6f} > theoretical upper {hi:.6f}"


def test_evaluate_bounds_grid_benchmark_support_promotes_to_bounds_supported() -> None:
    """Passing benchmark_support promotes eligible pairs to bounds_supported."""
    from sabench.analysis.bounds_grid import BENCHMARK_OUTPUT_BOUNDS

    results = evaluate_bounds_grid(
        benchmark_keys=("LinearModel",),
        transform_keys=("affine_a2_b1", "abs_pointwise"),
        n_base=64,
        seed=0,
        taylor_order=1,
        benchmark_support=BENCHMARK_OUTPUT_BOUNDS,
    )

    statuses = {r.transform_key: r.bounds_status for r in results}
    assert statuses["affine_a2_b1"] == "bounds_supported"
    # Non-smooth transforms remain non-applicable regardless of support
    assert statuses["abs_pointwise"] == "bounds_not_smooth"


def test_evaluate_bounds_grid_support_by_pair_takes_precedence() -> None:
    """Explicit support_by_pair overrides benchmark_support for the same pair."""
    from sabench.analysis.bounds_grid import BENCHMARK_OUTPUT_BOUNDS

    # Use an invalid support range for the pair; should cause domain_invalid
    # even though benchmark_support has the correct bounds.
    results = evaluate_bounds_grid(
        benchmark_keys=("LinearModel",),
        transform_keys=("affine_a2_b1",),
        n_base=32,
        seed=0,
        taylor_order=1,
        benchmark_support=BENCHMARK_OUTPUT_BOUNDS,
        support_by_pair={("LinearModel", "affine_a2_b1"): (100.0, 101.0)},
    )

    assert results[0].bounds_status == "bounds_domain_invalid"
