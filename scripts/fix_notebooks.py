#!/usr/bin/env python3
"""Script to implement all notebook cleanup changes."""

from __future__ import annotations

import json
from pathlib import Path


def load_nb(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_nb(nb: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def strip_outputs(nb: dict) -> dict:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def to_source(code: str) -> list[str]:
    """Convert a multiline string to notebook source list."""
    lines = code.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            if line:  # only add last line if non-empty
                result.append(line)
    return result


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─── 1. Fix demo.ipynb ───────────────────────────────────────────────────────

demo_path = REPO_ROOT / "notebooks" / "demo.ipynb"
nb = load_nb(str(demo_path))

# Cell 2: fix imports - add `import os` and OUTPUT_DIR block
cell2 = nb["cells"][2]
old_src = "".join(cell2["source"])

# Build new Cell 2 source
new_cell2 = """\
import json, pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sabench.analysis import jansen_s1_st
from sabench.benchmarks.functional import EpidemicSIR
from sabench.benchmarks.scalar import Ishigami
from sabench.benchmarks.spatial import Campbell2D
from sabench.sampling import saltelli_sample
from sabench.transforms import (
    CONCAVE_TRANSFORMS,
    CONVEX_TRANSFORMS,
    MONOTONE_TRANSFORMS,
    POINTWISE_TRANSFORMS,
    SMOOTH_TRANSFORMS,
    TRANSFORMS,
    apply_transform,
    score_transform,
)

_REPO_ROOT = pathlib.Path.cwd()
if _REPO_ROOT.name == "notebooks":
    _REPO_ROOT = _REPO_ROOT.parent
OUTPUT_DIR = pathlib.Path(os.environ.get("SABENCH_DEMO_OUTPUT_DIR", str(_REPO_ROOT / "outputs" / "demo")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
    }
)
print(f"sabench loaded. {len(TRANSFORMS)} transforms registered.")"""
cell2["source"] = to_source(new_cell2)

# Cell 6: fix savefig
cell6 = nb["cells"][6]
src6 = "".join(cell6["source"])
src6 = src6.replace(
    'plt.savefig("notebooks/scalar_demo.png", bbox_inches="tight", dpi=150)',
    'plt.savefig(OUTPUT_DIR / "scalar_demo.png", bbox_inches="tight", dpi=150)',
)
src6 = src6.replace(
    'print("Saved scalar_demo.png")', "print(f\"Saved {OUTPUT_DIR / 'scalar_demo.png'}\")"
)
cell6["source"] = to_source(src6)

# Cell 10: fix savefig
cell10 = nb["cells"][10]
src10 = "".join(cell10["source"])
src10 = src10.replace(
    'plt.savefig("notebooks/temporal_demo.png", bbox_inches="tight", dpi=150)',
    'plt.savefig(OUTPUT_DIR / "temporal_demo.png", bbox_inches="tight", dpi=150)',
)
src10 = src10.replace(
    'print("Saved temporal_demo.png")', "print(f\"Saved {OUTPUT_DIR / 'temporal_demo.png'}\")"
)
cell10["source"] = to_source(src10)

# Cell 14: fix savefig
cell14 = nb["cells"][14]
src14 = "".join(cell14["source"])
src14 = src14.replace(
    'plt.savefig("notebooks/spatial_demo.png", bbox_inches="tight", dpi=150)',
    'plt.savefig(OUTPUT_DIR / "spatial_demo.png", bbox_inches="tight", dpi=150)',
)
src14 = src14.replace(
    'print("Saved spatial_demo.png")', "print(f\"Saved {OUTPUT_DIR / 'spatial_demo.png'}\")"
)
cell14["source"] = to_source(src14)

# Cell 16: fix meta_path and savefig
cell16 = nb["cells"][16]
src16 = "".join(cell16["source"])
src16 = src16.replace(
    'meta_path = pathlib.Path("sabench/metadata/transforms_metadata.json")',
    'meta_path = _REPO_ROOT / "sabench" / "metadata" / "transforms_metadata.json"',
)
src16 = src16.replace(
    'plt.savefig("notebooks/transform_landscape.png", bbox_inches="tight", dpi=150)',
    'plt.savefig(OUTPUT_DIR / "transform_landscape.png", bbox_inches="tight", dpi=150)',
)
src16 = src16.replace(
    'print("Saved transform_landscape.png")',
    "print(f\"Saved {OUTPUT_DIR / 'transform_landscape.png'}\")",
)
cell16["source"] = to_source(src16)

strip_outputs(nb)
save_nb(nb, str(demo_path))
print("Fixed demo.ipynb")


# ─── 2. Rewrite noncommutativity_grid_analysis.ipynb ─────────────────────────

noncomm_path = REPO_ROOT / "notebooks" / "noncommutativity_grid_analysis.ipynb"
nb2 = load_nb(str(noncomm_path))

# Cell 2 (config cell) - replace source
config_source = """\
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from sabench.analysis.grid import evaluate_noncommutativity_grid
from sabench.benchmarks import BENCHMARK_REGISTRY
from sabench.transforms import TRANSFORM_REGISTRY

N_BASE = int(os.environ.get("SABENCH_GRID_N_BASE", "128"))
RNG_SEED = int(os.environ.get("SABENCH_GRID_SEED", "20260429"))
TAU = float(os.environ.get("SABENCH_GRID_TAU", "0.05"))
TOP_K = int(os.environ.get("SABENCH_GRID_TOP_K", "3"))
MAX_BENCHMARKS = int(os.environ.get("SABENCH_GRID_MAX_BENCHMARKS", "0"))
MAX_TRANSFORMS = int(os.environ.get("SABENCH_GRID_MAX_TRANSFORMS", "0"))

_HERE = Path.cwd()
_REPO_ROOT = _HERE.parent if _HERE.name == "notebooks" else _HERE
OUTPUT_DIR = Path(
    os.environ.get(
        "SABENCH_GRID_OUTPUT_DIR",
        str(_REPO_ROOT / "outputs" / "noncommutativity_grid_analysis"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

benchmark_keys = tuple(BENCHMARK_REGISTRY)
transform_keys = tuple(TRANSFORM_REGISTRY)
if MAX_BENCHMARKS > 0:
    benchmark_keys = benchmark_keys[:MAX_BENCHMARKS]
if MAX_TRANSFORMS > 0:
    transform_keys = transform_keys[:MAX_TRANSFORMS]

print(
    f"Grid: {len(benchmark_keys)} benchmarks × {len(transform_keys)} transforms, "
    f"N_BASE={N_BASE}, TAU={TAU}, seed={RNG_SEED}."
)"""
nb2["cells"][2]["source"] = to_source(config_source)

# Cell 4 (grid execution cell) - replace source
grid_source = """\
results = evaluate_noncommutativity_grid(
    benchmark_keys=benchmark_keys,
    transform_keys=transform_keys,
    n_base=N_BASE,
    seed=RNG_SEED,
    tau=TAU,
    top_k=TOP_K,
)
rows = [result.as_dict() for result in results]
df = pd.DataFrame(rows)
print(f"Evaluated {len(df)} candidate pairs")
print(df["metrics_status"].value_counts().to_string())"""
nb2["cells"][4]["source"] = to_source(grid_source)

# Cell 6 (export cell) - replace source
export_source = """\
PAIR_STATUS_COLUMNS = [
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
    "raw_output_shape",
    "transformed_output_shape",
    "raw_output_finite",
    "transformed_output_finite",
    "raw_variance",
    "transformed_variance",
]
METRIC_COLUMNS = [
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
    "top_driver_y_s1",
    "top_driver_z_s1",
    "top_driver_y_st",
    "top_driver_z_st",
]

df_status = df[[c for c in PAIR_STATUS_COLUMNS if c in df.columns]].copy()
df_metrics = df.loc[df["metrics_status"] == "computed", [c for c in METRIC_COLUMNS if c in df.columns]].reset_index(drop=True)

df_status.to_csv(OUTPUT_DIR / "pair_status.csv", index=False)
df_metrics.to_csv(OUTPUT_DIR / "noncommutativity_metrics.csv", index=False)

print(f"Wrote pair_status.csv ({len(df_status)} rows)")
print(f"Wrote noncommutativity_metrics.csv ({len(df_metrics)} rows)")
try:
    display(df_metrics.head(10)) if len(df_metrics) > 0 else print("No computed metric rows.")
except NameError:
    print(df_metrics.head(10).to_string()) if len(df_metrics) > 0 else print("No computed metric rows.")"""
nb2["cells"][6]["source"] = to_source(export_source)

# Cell 8 (summary cell) - replace source
summary_source = """\
SUMMARY_METRICS = ["D_s1", "delta_s1", "D_st", "delta_st"]

if not df_metrics.empty:
    available = [m for m in SUMMARY_METRICS if m in df_metrics.columns]

    df_by_transform = (
        df_metrics.groupby("transform_key")[available]
        .agg(["mean", "max"])
        .round(4)
    )
    df_by_transform.columns = [f"{agg}_{m}" for m, agg in df_by_transform.columns]
    df_by_transform = df_by_transform.reset_index()

    df_by_benchmark = (
        df_metrics.groupby("benchmark_key")[available]
        .agg(["mean", "max"])
        .round(4)
    )
    df_by_benchmark.columns = [f"{agg}_{m}" for m, agg in df_by_benchmark.columns]
    df_by_benchmark = df_by_benchmark.reset_index()

    df_by_transform.to_csv(OUTPUT_DIR / "summary_by_transform.csv", index=False)
    df_by_benchmark.to_csv(OUTPUT_DIR / "summary_by_benchmark.csv", index=False)

    print(f"Wrote summary_by_transform.csv ({len(df_by_transform)} rows)")
    print(f"Wrote summary_by_benchmark.csv ({len(df_by_benchmark)} rows)")
    print("\\nTop 10 pairs by first-order sensitivity shift (delta_s1):")
    if "delta_s1" in df_metrics.columns:
        try:
            display(
                df_metrics.nlargest(10, "delta_s1")[
                    ["benchmark_key", "transform_key", "D_s1", "delta_s1", "D_st", "delta_st"]
                ].reset_index(drop=True)
            )
        except NameError:
            print(
                df_metrics.nlargest(10, "delta_s1")[
                    ["benchmark_key", "transform_key", "D_s1", "delta_s1", "D_st", "delta_st"]
                ].reset_index(drop=True).to_string()
            )
else:
    df_by_transform = pd.DataFrame(columns=["transform_key"])
    df_by_benchmark = pd.DataFrame(columns=["benchmark_key"])
    df_by_transform.to_csv(OUTPUT_DIR / "summary_by_transform.csv", index=False)
    df_by_benchmark.to_csv(OUTPUT_DIR / "summary_by_benchmark.csv", index=False)
    print("No computed pairs — summary tables written as empty.")"""
nb2["cells"][8]["source"] = to_source(summary_source)

# Cell 10 (interpretation cell) - replace source
interp_source = """\
print("Pair status breakdown:")
print(df["pair_status"].value_counts().to_string())
print()
if not df_metrics.empty and "delta_s1" in df_metrics.columns:
    top = df_metrics.nlargest(1, "delta_s1").iloc[0]
    print(f"Largest first-order shift: {top['benchmark_key']} + {top['transform_key']}  delta_s1={top['delta_s1']:.4f}")
if not df_metrics.empty and "delta_st" in df_metrics.columns:
    top_st = df_metrics.nlargest(1, "delta_st").iloc[0]
    print(f"Largest total-effect shift: {top_st['benchmark_key']} + {top_st['transform_key']}  delta_st={top_st['delta_st']:.4f}")"""
nb2["cells"][10]["source"] = to_source(interp_source)

strip_outputs(nb2)
save_nb(nb2, str(noncomm_path))
print("Fixed noncommutativity_grid_analysis.ipynb")


# ─── 3. Rewrite bounds_theorem_grid_analysis.ipynb ───────────────────────────

bounds_path = REPO_ROOT / "notebooks" / "bounds_theorem_grid_analysis.ipynb"
nb3 = load_nb(str(bounds_path))

# Cell 3 (config cell) - replace source
bounds_config_source = """\
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from sabench.analysis.bounds import supported_smooth_pointwise_transform_keys
from sabench.analysis.bounds_grid import BOUNDS_STATUSES, evaluate_bounds_grid
from sabench.benchmarks import BENCHMARK_REGISTRY
from sabench.transforms import TRANSFORM_REGISTRY

N_BASE = int(os.environ.get("SABENCH_BOUNDS_N_BASE", "128"))
RNG_SEED = int(os.environ.get("SABENCH_BOUNDS_SEED", "20260429"))
TAYLOR_ORDER = int(os.environ.get("SABENCH_BOUNDS_TAYLOR_ORDER", "2"))
MAX_BENCHMARKS = int(os.environ.get("SABENCH_BOUNDS_MAX_BENCHMARKS", "0"))
MAX_TRANSFORMS = int(os.environ.get("SABENCH_BOUNDS_MAX_TRANSFORMS", "0"))

_HERE = Path.cwd()
_REPO_ROOT = _HERE.parent if _HERE.name == "notebooks" else _HERE
OUTPUT_DIR = Path(
    os.environ.get(
        "SABENCH_BOUNDS_OUTPUT_DIR",
        str(_REPO_ROOT / "outputs" / "bounds_theorem_grid_analysis"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

benchmark_keys = tuple(BENCHMARK_REGISTRY)
supported_transform_keys = tuple(
    key for key in supported_smooth_pointwise_transform_keys() if key in TRANSFORM_REGISTRY
)
remaining_transform_keys = tuple(
    key for key in TRANSFORM_REGISTRY if key not in set(supported_transform_keys)
)
transform_keys = supported_transform_keys + remaining_transform_keys

if MAX_BENCHMARKS > 0:
    benchmark_keys = benchmark_keys[:MAX_BENCHMARKS]
if MAX_TRANSFORMS > 0:
    transform_keys = transform_keys[:MAX_TRANSFORMS]

print(
    f"Bounds grid: {len(benchmark_keys)} benchmarks × {len(transform_keys)} transforms, "
    f"N_BASE={N_BASE}, Taylor order={TAYLOR_ORDER}."
)"""
nb3["cells"][3]["source"] = to_source(bounds_config_source)

# Cell 5 (grid execution cell) - replace source
bounds_grid_source = """\
results = evaluate_bounds_grid(
    benchmark_keys=benchmark_keys,
    transform_keys=transform_keys,
    n_base=N_BASE,
    seed=RNG_SEED,
    taylor_order=TAYLOR_ORDER,
)
rows = [result.as_dict() for result in results]
df_bounds = pd.DataFrame(rows)

print(f"Evaluated {len(df_bounds)} candidate pairs")
print(df_bounds["bounds_status"].value_counts().to_string())"""
nb3["cells"][5]["source"] = to_source(bounds_grid_source)

# Cell 7 (export cell) - replace source
bounds_export_source = """\
from collections import Counter

PAIR_STATUS_COLUMNS = [
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
    "output_shape",
    "output_finite",
    "output_variance",
    "taylor_order",
]
TAYLOR_COLUMNS = [
    "benchmark_key",
    "transform_key",
    "bounds_status",
    "support_source",
    "support_lower",
    "support_upper",
    "taylor_status",
    "eta_empirical",
    "eta_sufficient",
    "eta_empirical_lt_one",
    "eta_sufficient_lt_one",
    "projection_bound_s1_max",
    "projection_bound_st_max",
    "projection_bound_s1_mean",
    "projection_bound_st_mean",
    "reference_shift_s1_max",
    "reference_shift_st_max",
    "reference_shift_s1_mean",
    "reference_shift_st_mean",
]
LOCAL_AFFINE_COLUMNS = [
    "benchmark_key",
    "transform_key",
    "bounds_status",
    "local_affine_status",
    "support_source",
    "sigma_y",
    "mu4",
    "phi_prime_mu",
    "rho2",
    "kappa",
    "moment_ratio",
    "lambda_value",
    "eta_upper",
    "lambda_lt_two",
]

status_counts = Counter(df_bounds["bounds_status"])

df_pair_status = df_bounds[[c for c in PAIR_STATUS_COLUMNS if c in df_bounds.columns]].copy()
df_taylor = df_bounds.loc[
    df_bounds["support_source"].notna(), [c for c in TAYLOR_COLUMNS if c in df_bounds.columns]
].reset_index(drop=True)
df_local_affine = df_bounds.loc[
    df_bounds["local_affine_status"].notna(), [c for c in LOCAL_AFFINE_COLUMNS if c in df_bounds.columns]
].reset_index(drop=True)
df_summary = pd.DataFrame(
    [{"bounds_status": s, "count": status_counts.get(s, 0)} for s in BOUNDS_STATUSES]
)

df_pair_status.to_csv(OUTPUT_DIR / "bounds_pair_status.csv", index=False)
df_taylor.to_csv(OUTPUT_DIR / "taylor_reference_results.csv", index=False)
df_local_affine.to_csv(OUTPUT_DIR / "local_affine_results.csv", index=False)
df_summary.to_csv(OUTPUT_DIR / "bounds_summary.csv", index=False)

print(f"Wrote bounds_pair_status.csv ({len(df_pair_status)} rows)")
print(f"Wrote taylor_reference_results.csv ({len(df_taylor)} rows)")
print(f"Wrote local_affine_results.csv ({len(df_local_affine)} rows)")
print(f"Wrote bounds_summary.csv")
print()
print("Status summary:")
try:
    display(df_summary)
except NameError:
    print(df_summary.to_string(index=False))
if not df_taylor.empty:
    print("\\nTaylor reference diagnostics (first 10):")
    try:
        display(df_taylor.head(10))
    except NameError:
        print(df_taylor.head(10).to_string())"""
nb3["cells"][7]["source"] = to_source(bounds_export_source)

strip_outputs(nb3)
save_nb(nb3, str(bounds_path))
print("Fixed bounds_theorem_grid_analysis.ipynb")

print("All notebook changes applied.")

# This file is extended with fixes
