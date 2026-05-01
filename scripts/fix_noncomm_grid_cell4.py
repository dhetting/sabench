#!/usr/bin/env python3
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def to_source(code):
    lines = code.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            if line:
                result.append(line)
    return result


nb_path = REPO_ROOT / "notebooks" / "noncommutativity_grid_analysis.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

# Fix cell 4 - add column renaming
new_cell4 = """\
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

# Normalize column names to short forms used in exports and summaries
_RENAME = {
    "decision_score_s1": "D_s1",
    "decision_score_st": "D_st",
    "sensitivity_shift_s1": "delta_s1",
    "sensitivity_shift_st": "delta_st",
    "threshold_flip_count_s1": "threshold_flip_s1",
    "threshold_flip_count_st": "threshold_flip_st",
    "top_source_index_s1": "top_driver_y_s1",
    "top_source_index_st": "top_driver_y_st",
    "top_transformed_index_s1": "top_driver_z_s1",
    "top_transformed_index_st": "top_driver_z_st",
}
df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns})

print(f"Evaluated {len(df)} candidate pairs")
print(df["metrics_status"].value_counts().to_string())"""

nb["cells"][4]["source"] = to_source(new_cell4)

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Fixed cell 4 in noncommutativity_grid_analysis.ipynb")
