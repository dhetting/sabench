"""Notebook contract tests for the bounds-theorem grid analysis."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import sabench
from sabench.analysis.bounds_grid import BOUNDS_STATUSES

NOTEBOOK_NAME = "03_bounds_theorem_grid_analysis.ipynb"
OUTPUT_FILENAMES = {
    "bounds_pair_status.csv",
    "taylor_reference_results.csv",
    "local_affine_results.csv",
    "bounds_summary.csv",
}
PAIR_STATUS_COLUMNS = {
    "benchmark_key",
    "transform_key",
    "bounds_status",
    "reason",
    "benchmark_output_kind",
    "transform_mechanism",
    "transform_tags",
    "n_base",
    "seed",
    "n_inputs",
    "n_evaluations",
}
TAYLOR_COLUMNS = {
    "benchmark_key",
    "transform_key",
    "bounds_status",
    "support_source",
    "taylor_status",
    "eta_empirical",
    "eta_sufficient",
    "projection_bound_s1_max",
    "projection_bound_st_max",
    "reference_shift_s1_max",
    "reference_shift_st_max",
}
LOCAL_AFFINE_COLUMNS = {
    "benchmark_key",
    "transform_key",
    "bounds_status",
    "local_affine_status",
    "lambda_value",
    "eta_upper",
    "lambda_lt_two",
}


def _repo_root() -> Path:
    return Path(sabench.__file__).resolve().parent.parent


def _notebook_path() -> Path:
    return _repo_root() / "notebooks" / NOTEBOOK_NAME


def _notebook() -> dict[str, object]:
    return json.loads(_notebook_path().read_text(encoding="utf-8"))


def _code_cells() -> list[str]:
    notebook = _notebook()
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def _notebook_text() -> str:
    return _notebook_path().read_text(encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_bounds_theorem_grid_notebook_exists_and_is_clean() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4

    cells = notebook["cells"]
    assert cells
    for cell in cells:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_bounds_theorem_grid_notebook_uses_registry_grid_engine() -> None:
    text = _notebook_text()

    assert "evaluate_bounds_grid" in text
    assert "BENCHMARK_REGISTRY" in text
    assert "TRANSFORM_REGISTRY" in text
    assert "SABENCH_BOUNDS_MAX_BENCHMARKS" in text
    assert "SABENCH_BOUNDS_MAX_TRANSFORMS" in text
    assert "SABENCH_BOUNDS_N_BASE" in text
    assert "bounds_memo_v22.tex" in text
    assert "sample-range diagnostics" in text
    assert "V_m" in text


def test_bounds_theorem_grid_notebook_declares_expected_exports_and_statuses() -> None:
    text = _notebook_text()
    for filename in OUTPUT_FILENAMES:
        assert filename in text
    for status in BOUNDS_STATUSES:
        assert status in text


def test_bounds_theorem_grid_notebook_executes_smoke_grid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "outputs"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SABENCH_BOUNDS_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("SABENCH_BOUNDS_N_BASE", "32")
    monkeypatch.setenv("SABENCH_BOUNDS_MAX_BENCHMARKS", "2")
    monkeypatch.setenv("SABENCH_BOUNDS_MAX_TRANSFORMS", "6")
    monkeypatch.setenv("SABENCH_BOUNDS_SEED", "20260429")
    monkeypatch.setenv("SABENCH_BOUNDS_TAYLOR_ORDER", "1")

    namespace: dict[str, object] = {"__name__": "__main__"}
    for source in _code_cells():
        exec(compile(source, NOTEBOOK_NAME, "exec"), namespace)

    for filename in OUTPUT_FILENAMES:
        assert (output_dir / filename).is_file()

    pair_rows = _read_csv(output_dir / "bounds_pair_status.csv")
    taylor_rows = _read_csv(output_dir / "taylor_reference_results.csv")
    local_affine_rows = _read_csv(output_dir / "local_affine_results.csv")
    summary_rows = _read_csv(output_dir / "bounds_summary.csv")

    assert pair_rows
    assert set(pair_rows[0]) >= PAIR_STATUS_COLUMNS
    assert {row["bounds_status"] for row in pair_rows} <= set(BOUNDS_STATUSES)
    assert "bounds_diagnostic_sample_support" in {row["bounds_status"] for row in pair_rows}

    if taylor_rows:
        assert set(taylor_rows[0]) >= TAYLOR_COLUMNS
    if local_affine_rows:
        assert set(local_affine_rows[0]) >= LOCAL_AFFINE_COLUMNS

    assert summary_rows
    assert {"bounds_status", "count"} <= set(summary_rows[0])


def test_bounds_theorem_grid_notebook_does_not_modify_environment() -> None:
    before = dict(os.environ)
    _code_cells()
    assert dict(os.environ) == before
