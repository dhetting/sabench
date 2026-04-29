"""Notebook contract tests for the noncommutativity grid analysis."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import sabench

NOTEBOOK_NAME = "02_noncommutativity_grid_analysis.ipynb"
OUTPUT_FILENAMES = {
    "pair_status.csv",
    "noncommutativity_metrics.csv",
    "summary_by_transform.csv",
    "summary_by_benchmark.csv",
}
PAIR_STATUS_COLUMNS = {
    "benchmark_key",
    "transform_key",
    "pair_status",
    "metrics_status",
    "reason",
    "benchmark_output_kind",
    "transform_mechanism",
    "transform_tags",
    "n_base",
    "seed",
    "n_inputs",
    "n_evaluations",
}
METRIC_COLUMNS = {
    "benchmark_key",
    "transform_key",
    "D_s1",
    "delta_s1",
    "D_st",
    "delta_st",
    "threshold_flip_s1",
    "threshold_flip_st",
    "topk_changed_s1",
    "topk_changed_st",
    "max_abs_shift_s1",
    "max_abs_shift_st",
    "mean_abs_shift_s1",
    "mean_abs_shift_st",
    "spearman_s1",
    "spearman_st",
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


def test_noncommutativity_grid_notebook_exists_and_is_clean() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4

    cells = notebook["cells"]
    assert cells
    for cell in cells:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_noncommutativity_grid_notebook_uses_registry_grid_engine() -> None:
    text = _notebook_text()

    assert "evaluate_noncommutativity_grid" in text
    assert "BENCHMARK_REGISTRY" in text
    assert "TRANSFORM_REGISTRY" in text
    assert "SABENCH_GRID_MAX_BENCHMARKS" in text
    assert "SABENCH_GRID_MAX_TRANSFORMS" in text
    assert "SABENCH_GRID_N_BASE" in text
    assert "D_s1" in text
    assert "delta_s1" in text
    assert "noncommutativity_detailed.tex" in text


def test_noncommutativity_grid_notebook_declares_expected_exports() -> None:
    text = _notebook_text()
    for filename in OUTPUT_FILENAMES:
        assert filename in text


def test_noncommutativity_grid_notebook_executes_smoke_grid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "outputs"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SABENCH_GRID_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("SABENCH_GRID_N_BASE", "32")
    monkeypatch.setenv("SABENCH_GRID_MAX_BENCHMARKS", "2")
    monkeypatch.setenv("SABENCH_GRID_MAX_TRANSFORMS", "4")
    monkeypatch.setenv("SABENCH_GRID_SEED", "20260429")

    namespace: dict[str, object] = {"__name__": "__main__"}
    for source in _code_cells():
        exec(compile(source, NOTEBOOK_NAME, "exec"), namespace)

    for filename in OUTPUT_FILENAMES:
        assert (output_dir / filename).is_file()

    pair_rows = _read_csv(output_dir / "pair_status.csv")
    metric_rows = _read_csv(output_dir / "noncommutativity_metrics.csv")
    summary_by_transform = _read_csv(output_dir / "summary_by_transform.csv")
    summary_by_benchmark = _read_csv(output_dir / "summary_by_benchmark.csv")

    assert pair_rows
    assert set(pair_rows[0]) >= PAIR_STATUS_COLUMNS
    assert {row["metrics_status"] for row in pair_rows}

    if metric_rows:
        assert set(metric_rows[0]) >= METRIC_COLUMNS
        for row in metric_rows:
            for metric in ("D_s1", "delta_s1", "D_st", "delta_st"):
                value = float(row[metric])
                assert 0.0 <= value <= 1.0

    assert summary_by_transform or not metric_rows
    assert summary_by_benchmark or not metric_rows


def test_noncommutativity_grid_notebook_does_not_modify_environment() -> None:
    before = dict(os.environ)
    _code_cells()
    assert dict(os.environ) == before
