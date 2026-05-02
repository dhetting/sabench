#!/usr/bin/env python3
"""
Rewrite the markdown documentation cells in the two analysis notebooks.

Usage:
    python scripts/rewrite_notebook_docs.py

All code cells are preserved byte-for-byte. Only markdown cells are replaced
or inserted. New markdown cells may be added before or between existing cells.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

# ---------------------------------------------------------------------------
# Noncommutativity notebook
# ---------------------------------------------------------------------------

NONCOMM_CELLS: list[dict[str, str | list[str]]] = []
# Each entry is either {"type": "code", "source": <original source string>}
# or {"type": "markdown", "source": <new source string>}


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source.strip())


def preserve_code(nb: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    """Return only the code cells from the notebook in order."""
    return [c for c in nb.cells if c.cell_type == "code"]


# ---------------------------------------------------------------------------
# Noncommutativity notebook content
# ---------------------------------------------------------------------------

NONCOMM_INTRO = """\
# Noncommutativity grid analysis

**Purpose.** This notebook runs a systematic empirical sweep to measure how
much output post-processing transformations change the variance-based (Sobol)
sensitivity profile of a model.  It implements the scoring metrics from the
companion paper:

> Hettinger, D. (2026). *Non-Commutativity of Sobol Sensitivity Indices Under
> Output Transformations of Functional, Spatial, and Scalar Models.*
> Prepared for the *Electronic Journal of Statistics.*
> Source: `docs/manuscripts/noncommutativity_detailed.tex`.

The scientific question is: if you apply a transformation $\\phi$ to a model
output before computing Sobol indices, do you get the same result as computing
Sobol indices first and then summarising?  Mathematically, does
$S_i(\\phi \\circ f) = S_i(f)$?  The answer is **no** for any nonlinear
$\\phi$ (Proposition 1).  This notebook quantifies *how large* the discrepancy
is across a registry of benchmarks and transformations.

**What this notebook produces.**
Running all cells creates four CSV files in
`outputs/noncommutativity_grid_analysis/` (or the directory set by
`SABENCH_GRID_OUTPUT_DIR`):

| File | Contents |
|------|----------|
| `pair_status.csv` | One row per (benchmark, transform) pair; status, metadata, finite-ness checks |
| `noncommutativity_metrics.csv` | Computed metrics for pairs that passed all checks |
| `summary_by_transform.csv` | Mean and max of key metrics, grouped by transform |
| `summary_by_benchmark.csv` | Mean and max of key metrics, grouped by benchmark |

**How to run.**
```bash
# Fast smoke run (default — CI-safe, small N)
jupyter notebook notebooks/noncommutativity_grid_analysis.ipynb

# Full run (all pairs, larger sample)
SABENCH_GRID_N_BASE=1024 jupyter notebook notebooks/noncommutativity_grid_analysis.ipynb

# Tiny debug run (first 3 benchmarks, first 10 transforms)
SABENCH_GRID_MAX_BENCHMARKS=3 SABENCH_GRID_MAX_TRANSFORMS=10 \\
  jupyter notebook notebooks/noncommutativity_grid_analysis.ipynb
```

**Dependencies.**  All scientific logic is in tested `sabench` package modules;
this notebook only orchestrates and exports results.  No notebook-local
scientific computations.
"""

NONCOMM_BACKGROUND_SOBOL = """\
## Background: Sobol sensitivity analysis

Variance-based (Sobol) global sensitivity analysis decomposes the total output
variance of a model $Y = f(\\mathbf{X})$ — with independent, uniformly
distributed inputs $X_1, \\ldots, X_d$ — according to the contribution of each
input and each interaction.

**First-order Sobol index** for input $X_i$:

$$S_i = \\frac{D_i}{\\mathrm{Var}[Y]}, \\qquad
  D_i = \\mathrm{Var}_{X_i}\\!\\bigl[\\mathbb{E}[Y \\mid X_i]\\bigr].$$

$S_i$ measures the fraction of total output variance explained by $X_i$ alone,
averaging over all other inputs.

**Total-effect Sobol index**:

$$S_i^T = 1 - \\frac{\\mathrm{Var}_{\\mathbf{X}_{\\sim i}}\\!\\bigl[\\mathbb{E}[Y \\mid \\mathbf{X}_{\\sim i}]\\bigr]}{\\mathrm{Var}[Y]},$$

where $\\mathbf{X}_{\\sim i}$ is all inputs except $X_i$.  $S_i^T$ measures the
total contribution of $X_i$, including all interactions.

**Interpretation.** If $S_i^T < \\tau$ for some threshold $\\tau$ (e.g.,
$\\tau = 0.05$), input $X_i$ can be fixed at any value without materially
reducing output variance — the *factor-fixing* decision.  Whether that decision
changes when the output is transformed is the central question of this notebook.

**Estimation.** This notebook uses the Jansen (1999) estimator with the
Saltelli (2002) input design, requiring $N(d+2)$ model evaluations for $d$
inputs and $N$ base samples.  Estimated indices are clipped to $[0, 1]$ to
handle finite-sample noise.  For benchmarks with multi-dimensional outputs
(e.g., spatial grids, temporal trajectories), the per-input index is
aggregated as a variance-weighted mean across output dimensions:

$$\\hat{S}_i = \\frac{\\sum_j \\mathrm{Var}[Y_j] \\cdot \\hat{S}_{ij}}{\\sum_j \\mathrm{Var}[Y_j]},$$

clipped to $[0, 1]$.  This gives a single sensitivity *profile*
$(\\hat{S}_1, \\ldots, \\hat{S}_d)$ per pair regardless of output dimensionality.
"""

NONCOMM_THEOREM = """\
## The commutativity theorem and its consequences

**Proposition 1 (Commutativity iff affine).** A continuous nonconstant
pointwise transform $\\phi : \\mathbb{R} \\to \\mathbb{R}$ satisfies
$S_i(\\phi \\circ f) = S_i(f)$ for every admissible square-integrable model $f$
if and only if $\\phi$ is affine (i.e., $\\phi(y) = ay + b$, $a \\neq 0$).

*Consequence for practitioners:* **every nonlinear transformation of model
output can change the Sobol sensitivity profile.**  Log transforms, power laws,
exceedance probabilities, block averages, cumulative sums of nonlinear functions
— none of these preserve Sobol indices in general.  The magnitude of the change
is empirically large for environmental and engineering transforms
($\\bar{D} \\approx 0.37$, $\\bar{\\Delta} \\approx 0.60$) and smaller but
nonzero for temporal/linear operators.

**Proposition 3 (Scalar invariance under monotone transforms).** Strictly
monotone scalar transforms produce identical Decision Score $D$ on any scalar
benchmark.  The Sensitivity Shift $\\Delta$ may still be nonzero.  This result
applies to scalar-output models only; spatial and temporal benchmarks are not
covered.  *Practical implication:* rank transforms and log transforms of
scalar outputs do not change which inputs are classified as active or inactive
by the factor-fixing threshold — but they do change the quantitative index
values.

**Change-of-support (Proposition 2).** The change-of-support (COS) operator
that block-averages a spatial output commutes with Sobol index computation if
and only if the spatial sensitivity structure is homogeneous.  Heterogeneous
spatial benchmarks (Campbell3D) show $13\\times$ amplified COS noncommutativity
relative to homogeneous ones.

*All three propositions are proved in the companion paper and in its appendix
for total-effect indices.*
"""

NONCOMM_METRICS = """\
## Metric definitions

### Decision Score $D$

The Decision Score measures the classification impact of the transformation on
the factor-fixing decision at threshold $\\tau$ (default $\\tau = 0.05$).

Define the soft activity indicator using a sigmoid centred at $\\tau$:

$$\\sigma(s; \\tau) = \\frac{1}{1 + \\exp\\!\\left(-\\frac{s - \\tau}{\\tau}\\right)}.$$

The Decision Score is then:

$$D = \\frac{1}{d} \\sum_{i=1}^{d} \\bigl| \\sigma(\\hat{S}_i(Z); \\tau) \\;-\\; \\sigma(\\hat{S}_i(Y); \\tau) \\bigr|,$$

where $Y$ is the raw benchmark output and $Z = \\phi(Y)$ is the transformed
output.

- $D = 0$: the transformation had **no** effect on factor-fixing classification
  (all inputs cross the threshold the same way before and after).
- $D = 1$: maximal classification reversal for every input.
- $D \\in [0, 1]$ and is directly comparable across benchmarks and transforms.
- The sigmoid width equals $\\tau$, calibrating sensitivity around the threshold
  boundary.  Unlike a rank correlation, $D$ responds to absolute index changes
  even when rankings are preserved.

### Sensitivity Shift $\\Delta$ (Bray-Curtis)

The Sensitivity Shift measures the raw redistribution of sensitivity mass,
using the Bray-Curtis dissimilarity:

$$\\Delta = \\frac{\\sum_{i=1}^{d} |\\hat{S}_i(Z) - \\hat{S}_i(Y)|}{\\sum_{i=1}^{d} \\hat{S}_i(Z) + \\sum_{i=1}^{d} \\hat{S}_i(Y)}.$$

- $\\Delta = 0$: the transform left the profile entirely intact.
- $\\Delta = 1$: the transform completely replaced one sensitivity profile with
  a non-overlapping one.
- The pooled denominator prevents instability when $\\hat{S}_i(Y) \\approx 0$.
- Bray-Curtis dissimilarity is the standard measure of community composition
  change in ecology, repurposed here for sensitivity profiles.

### Auxiliary metrics

| Column | Definition |
|--------|-----------|
| `threshold_flip_s1` / `threshold_flip_st` | Number of inputs that crossed the hard threshold $\\tau$ in either direction; integer count |
| `topk_changed_s1` / `topk_changed_st` | Boolean: whether the unordered top-$k$ most-influential input set changed |
| `max_abs_shift_s1` / `max_abs_shift_st` | $\\max_i |\\hat{S}_i(Z) - \\hat{S}_i(Y)|$ |
| `mean_abs_shift_s1` / `mean_abs_shift_st` | $\\frac{1}{d} \\sum_i |\\hat{S}_i(Z) - \\hat{S}_i(Y)|$ |
| `spearman_s1` / `spearman_st` | Spearman rank correlation between raw and transformed profiles |
| `top_driver_y_s1` / `top_driver_y_st` | 0-based index of the top-ranked input in the **raw** profile |
| `top_driver_z_s1` / `top_driver_z_st` | 0-based index of the top-ranked input in the **transformed** profile |

*Both first-order (`_s1`) and total-effect (`_st`) metrics are reported.*
"""

NONCOMM_CONFIG = """\
## Configuration

The configuration cell below sets all parameters for this run.  Override any
of these by setting the corresponding environment variable before launching
Jupyter, or by editing the cell directly.

| Parameter | Env var | Default | Meaning |
|-----------|---------|---------|---------|
| `N_BASE` | `SABENCH_GRID_N_BASE` | 128 | Base sample size for Saltelli design; actual evaluations = $N(d+2)$.  Use 512–2048 for publication-quality results. |
| `RNG_SEED` | `SABENCH_GRID_SEED` | 20260429 | Random seed for reproducibility.  Fixed across all pairs. |
| `TAU` | `SABENCH_GRID_TAU` | 0.05 | Factor-fixing threshold; 0.05 is the standard sensitivity-analysis convention (Saltelli et al., 2008). |
| `TOP_K` | `SABENCH_GRID_TOP_K` | 3 | Top-$k$ driver set size for `topk_changed` metric. |
| `MAX_BENCHMARKS` | `SABENCH_GRID_MAX_BENCHMARKS` | 0 (all) | Truncate benchmark list to first $n$ entries.  0 = use all. |
| `MAX_TRANSFORMS` | `SABENCH_GRID_MAX_TRANSFORMS` | 0 (all) | Truncate transform list to first $n$ entries.  0 = use all. |
| `OUTPUT_DIR` | `SABENCH_GRID_OUTPUT_DIR` | `outputs/noncommutativity_grid_analysis` | Directory for exported CSVs.  Created if absent. |

**CI and smoke runs.** The default `N_BASE=128` is intentionally small for fast
execution.  For a thorough run over the full registry ($\\approx 30$ benchmarks
$\\times$ $\\approx 117$ transforms), use `N_BASE=512` or larger and allow
several minutes of wall time.

**Reproducibility.** The `RNG_SEED` is applied globally; all pairs use the same
Saltelli sample reuse design, making results fully reproducible from the seed.
"""

NONCOMM_GRID_EXEC = """\
## Execute the registry-driven grid

The `evaluate_noncommutativity_grid()` function evaluates every
(benchmark, transform) pair and returns a list of result objects.  Each result
captures:

1. A **compatibility check** — whether the transform supports the benchmark's
   output kind (scalar, spatial, temporal).  Incompatible pairs are retained
   as `excluded` rows.
2. A **benchmark evaluation** — `N(d+2)` calls to `benchmark.evaluate()` using
   the Saltelli sampling design.
3. **Sobol index estimation** — Jansen (1999) estimator for both first-order
   $\\hat{S}_i$ and total-effect $\\hat{S}_i^T$, clipped to $[0, 1]$.
4. For multi-output benchmarks, **variance-weighted profile aggregation**
   produces one $d$-vector per index type.
5. **Metric computation** — $D$, $\\Delta$, flip counts, rank correlation, etc.
6. Pair-level exceptions are caught and recorded as structured failure rows,
   not crashes.

After the cell runs, `df` contains one row per pair.  The `pair_status` column
records the coarse compatibility outcome; the `metrics_status` column records
whether metrics were computed.

**`pair_status` values:**
- `included` — the pair was compatibility-checked and all pipeline steps were
  attempted; metric columns may or may not be populated (check `metrics_status`)
- `excluded` — the transform does not support the benchmark's output kind;
  no computation was performed

**`metrics_status` values:**
- `computed` — metrics are present and valid
- `not_applicable` — pair was `excluded`; no metrics attempted
- `failed_raw_evaluation` — benchmark evaluation raised an error
- `failed_raw_output_validation` — raw output had bad shape or non-finite values
- `failed_transform_evaluation` — transform raised an error
- `failed_transformed_output_validation` — transformed output had bad shape or
  non-finite values
- `failed_sobol_estimation` — Sobol estimation failed (e.g., zero variance)
- `failed_metric_computation` — Sobol estimates were computed but the resulting
  profiles contained non-finite values (e.g., due to numerical overflow when
  squaring very large transform outputs in the Jansen estimator)
"""

NONCOMM_COLUMN_GLOSSARY = """\
## Exported tables and column glossary

### `pair_status.csv`

One row per (benchmark, transform) pair, regardless of outcome.

| Column | Description |
|--------|-------------|
| `benchmark_key` | Registry key, e.g. `"Ishigami"` |
| `transform_key` | Registry key, e.g. `"log_shift_pos"` |
| `pair_status` | Coarse outcome: `included` (computation attempted) or `excluded` (incompatible) |
| `metrics_status` | Fine-grained outcome: `computed`, `not_applicable`, `failed_raw_evaluation`, `failed_raw_output_validation`, `failed_transform_evaluation`, `failed_transformed_output_validation`, `failed_sobol_estimation`, `failed_metric_computation` |
| `reason` | Human-readable reason string for non-`computed` rows |
| `benchmark_output_kind` | `scalar`, `spatial`, or `temporal` |
| `transform_mechanism` | `pointwise`, `spatial`, `temporal`, etc. |
| `transform_tags` | Semicolon-separated transform tags from registry |
| `n_base` | $N$ used in the Saltelli design |
| `seed` | Random seed |
| `n_inputs` | $d$, number of model inputs |
| `n_evaluations` | Total model calls $= N(d+2)$ |
| `raw_output_shape` | Shape of raw benchmark output matrix |
| `transformed_output_shape` | Shape after transformation |
| `raw_output_finite` | Whether all raw outputs were finite |
| `transformed_output_finite` | Whether all transformed outputs were finite |
| `raw_variance` | Total variance of raw output (scalar or aggregate) |
| `transformed_variance` | Total variance of transformed output |

### `noncommutativity_metrics.csv`

Only rows where `metrics_status == "computed"`.

| Column | Description |
|--------|-------------|
| `D_s1`, `D_st` | Decision Score for first-order and total-effect indices |
| `delta_s1`, `delta_st` | Bray-Curtis Sensitivity Shift |
| `threshold_flip_s1`, `threshold_flip_st` | Hard-threshold crossing count |
| `topk_changed_s1`, `topk_changed_st` | Whether top-$k$ driver set changed |
| `max_abs_shift_s1`, `max_abs_shift_st` | Maximum per-input absolute shift |
| `mean_abs_shift_s1`, `mean_abs_shift_st` | Mean per-input absolute shift |
| `spearman_s1`, `spearman_st` | Spearman rank correlation of profiles |
| `top_driver_y_s1`, `top_driver_y_st` | Top input index in **raw** profile |
| `top_driver_z_s1`, `top_driver_z_st` | Top input index in **transformed** profile |

### `summary_by_transform.csv` and `summary_by_benchmark.csv`

Aggregate statistics (mean and max) of the four key metrics grouped by
transform and by benchmark, respectively.  Useful for ranking which
transforms or benchmarks show the largest empirical noncommutativity.
"""

NONCOMM_SUMMARIES = """\
## Summaries by transform and benchmark

These tables aggregate the four headline metrics —
$D_{s1}$, $\\Delta_{s1}$, $D_{st}$, $\\Delta_{st}$ — grouped by transform and
by benchmark.  Both mean and max are reported.

- **Mean** reflects the typical noncommutativity magnitude of a given
  transform across all benchmarks it ran on, or of a given benchmark across
  all transforms.
- **Max** identifies the most extreme single pair for that transform or
  benchmark.

High mean $D$ or $\\Delta$ for a transform indicates it is systematically
disruptive to sensitivity rankings, regardless of the model.  High max with
low mean indicates a single benchmark is particularly sensitive to that
transform (e.g., near-threshold inputs that flip).
"""

NONCOMM_INTERPRETATION = """\
## Interpreting results

### What the scores mean

| Score range | $D$ (Decision Score) | $\\Delta$ (Sensitivity Shift) |
|-------------|---------------------|------------------------------|
| $\\approx 0$ | Transform has no effect on factor-fixing classification | Transform leaves sensitivity profile intact |
| $0.05$–$0.15$ | Small, possibly negligible classification perturbation | Modest redistribution of sensitivity mass |
| $0.15$–$0.30$ | Noticeable; some inputs cross the threshold | Material profile change |
| $> 0.30$ | Large; classification decisions would differ substantially | Dominant redistribution; profile is materially altered |

*These ranges are calibrated from empirical findings in the companion paper
and are indicative, not prescriptive.*

### Expected patterns from the companion paper

1. **Affine transforms** (e.g., `scale`, `shift`, `affine_linear`): $D = 0$
   exactly (to numerical precision).  This is the commutativity theorem.

2. **Strictly monotone scalar transforms** (e.g., `log_shift_pos`, `exp`,
   `sqrt_shift_pos`, `cbrt`, `arctan`, `tanh`): $D = 0$ on scalar benchmarks
   (Proposition 3).  $\\Delta$ may be nonzero.  These transforms change the
   quantitative Sobol index values but do not flip any factor-fixing
   classifications.

3. **Environmental and engineering transforms** (e.g., flood thresholds, power
   laws, exceedance probabilities): mean $D \\approx 0.36$–$0.37$ and mean
   $\\Delta \\approx 0.60$–$0.61$ in the companion paper.

4. **Temporal transforms** (cumulative sum, peak, exceedance duration): lower
   on average ($\\bar{D} \\approx 0.18$, $\\bar{\\Delta} \\approx 0.39$).  Linear
   cumulative sum on the Damped Oscillator scores near zero, confirming the
   theorem.

5. **Spatial COS transforms** on Campbell3D: mean $D$ is $13\\times$ higher than
   on the homogeneous Campbell2D benchmark.

### Sampling noise

With `N_BASE=128`, estimated Sobol indices have nontrivial sampling variance.
Small non-zero $D$ or $\\Delta$ values (e.g., $< 0.01$) should not be
interpreted as evidence of noncommutativity — they may reflect estimator noise.
Increase `N_BASE` to 512 or 1024 to reduce noise before drawing conclusions
about near-zero metrics.

### When `metrics_status` is not `computed`

- `not_applicable`: the pair was `excluded` (incompatible transform/benchmark
  output kind).  This is expected, not an error.
- `failed_raw_evaluation` / `failed_transform_evaluation`: an error was raised
  during benchmark or transform evaluation.  Inspect the `reason` column.
- `failed_raw_output_validation` / `failed_transformed_output_validation`:
  the output array had unexpected shape or non-finite values after evaluation.
- `failed_sobol_estimation`: Sobol estimation raised an error, usually due to
  zero raw variance (the benchmark's output was constant over the sample).
- `failed_metric_computation`: Sobol estimates were produced but the variance-
  weighted profiles contained non-finite values.  The most common cause is
  numerical overflow: transforms like `exp(s·y²)` that yield values near
  float64 max (~1.8×10³⁰⁸) overflow when squared during variance estimation
  inside the Jansen estimator.  The transform output itself may be finite, yet
  squaring produces infinity.  This is expected for pathological transform/
  benchmark combinations and is captured as a structured status, not a crash.
"""

# ---------------------------------------------------------------------------
# Bounds notebook content
# ---------------------------------------------------------------------------

BOUNDS_INTRO = """\
# Bounds-theorem grid analysis

**Purpose.** This notebook evaluates theorem-backed quantitative bounds on
Sobol index shifts under smooth output transformations, following the
theoretical framework of:

> Hettinger, D. (2026). *Quantitative Bounds on Sobol Index Shifts Under
> Smooth Output Transformations.* (Companion memo to the noncommutativity
> paper.)  Source: `docs/manuscripts/bounds_memo_v22.tex`.

The companion commutativity analysis (notebook
`noncommutativity_grid_analysis.ipynb`) asks *whether* a transform changes
Sobol indices.  This notebook asks *how much* it can change them, using a
mathematically rigorous perturbation bound.

**Critical interpretation note.** Bounds in this notebook compare the Sobol
indices of the transformed output $Z = \\phi(Y)$ to the Sobol indices of the
**Taylor reference** $V_m$ — not directly to those of $Y$.  This is the
correct object of comparison from the theorem's perspective; see the
*Mathematical framework* section below for why.

**What this notebook produces.**
Running all cells creates four CSV files in
`outputs/bounds_theorem_grid_analysis/` (or the directory set by
`SABENCH_BOUNDS_OUTPUT_DIR`):

| File | Contents |
|------|----------|
| `bounds_pair_status.csv` | One row per (benchmark, transform) pair; status and metadata |
| `taylor_reference_results.csv` | Taylor-reference diagnostics for eligible pairs |
| `local_affine_results.csv` | Local-affine ($m=1$) diagnostics for eligible pairs |
| `bounds_summary.csv` | Count of each status across the full grid |

**How to run.**
```bash
# Fast smoke run (default — CI-safe, small N)
jupyter notebook notebooks/bounds_theorem_grid_analysis.ipynb

# Full run (all compatible pairs, larger sample)
SABENCH_BOUNDS_N_BASE=1024 jupyter notebook notebooks/bounds_theorem_grid_analysis.ipynb

# Limit scope for testing
SABENCH_BOUNDS_MAX_BENCHMARKS=5 SABENCH_BOUNDS_MAX_TRANSFORMS=20 \\
  jupyter notebook notebooks/bounds_theorem_grid_analysis.ipynb
```

**Dependencies.** All scientific logic is in tested `sabench.analysis.bounds`
and `sabench.analysis.bounds_grid` modules.  No notebook-local scientific
computations.
"""

BOUNDS_BACKGROUND = """\
## Background: the quantitative perturbation question

The companion noncommutativity analysis (Proposition 1 of the paper) proves a
**qualitative** statement: only affine transforms commute with Sobol index
computation for every admissible model.  Every nonlinear transform can change
the sensitivity profile.

This notebook addresses the complementary **quantitative** question:

> For a specific smooth transform $\\phi$ applied to a specific benchmark, how
> large a Sobol index shift is possible, given the output distribution of that
> benchmark?

The answer uses a perturbation bound.  Instead of comparing $\\phi(Y)$ directly
to $Y$, the theorem compares $\\phi(Y)$ to the best polynomial approximation of
$\\phi(Y)$ in the vicinity of the output mean — the **Taylor reference** $V_m$.
This choice is natural because:

1. When $\\phi$ is exactly a degree-$m$ polynomial, the Taylor residual is zero
   and the bound is exact.
2. When $\\phi'(\\mu_Y) \\neq 0$ (nonzero slope at the mean), the local-affine
   reference $V_1 = \\phi'(\\mu_Y)(Y - \\mu_Y)$ has the same Sobol structure as
   $Y$, so bounding the distance from $\\phi(Y)$ to $V_1$ gives an indirect
   bound on the distance from $\\phi(Y)$ to $Y$.
3. When $\\phi'(\\mu_Y) = 0$ (a critical point), comparing $\\phi(Y)$ directly
   to $Y$ may be misleading because $Y$ and $\\phi(Y)$ may have structurally
   different sensitivity profiles regardless of how smooth $\\phi$ is.  The
   Taylor reference with $m=2$ gives the appropriate comparator.

**Scope of this notebook.** Only smooth (i.e., several-times differentiable)
pointwise transforms have registered derivative metadata in the `sabench`
registry; the bounds computation is skipped for nonsmooth, non-pointwise, or
multi-output-benchmark pairs.  The grid covers all registered smooth+pointwise
transforms applied to all registered benchmarks.
"""

BOUNDS_TAYLOR = """\
## Mathematical framework: Taylor reference and residual

### Setup

Let $Y = f(\\mathbf{X})$ be a scalar model output and $\\phi : \\mathbb{R} \\to
\\mathbb{R}$ a smooth transform.  Write $Z = \\phi(Y)$ and $\\mu_Y = \\mathbb{E}[Y]$.

### Taylor reference $V_m$

The order-$m$ Taylor reference is the $m$-th degree Taylor polynomial of
$\\phi$ expanded around $\\mu_Y$, applied to the centered output:

$$V_m = \\sum_{k=1}^{m} \\frac{\\phi^{(k)}(\\mu_Y)}{k!} (Y - \\mu_Y)^k.$$

Note: $V_m$ does *not* include the constant term $\\phi(\\mu_Y)$; it is the
centered, polynomial-approximated part of $\\phi(Y)$.

### Taylor residual $R_m$

$$R_m = \\phi(Y) - \\phi(\\mu_Y) - V_m.$$

$R_m$ is the error of the Taylor approximation, shifted to have the same mean
as $V_m$ (both are zero-mean to the extent $\\phi$ is well-approximated).

### Empirical $\\eta_m$

The key dimensionless quantity is the ratio of the residual's standard
deviation to the reference's standard deviation:

$$\\eta_m = \\frac{\\mathrm{sd}(R_m - \\mathbb{E}[R_m])}{\\mathrm{sd}(V_m)}.$$

- $\\eta_m = 0$: $\\phi$ is exactly a degree-$m$ polynomial; the bound is
  exact with zero error.
- $\\eta_m < 1$: the bound is finite and meaningful.
- $\\eta_m \\geq 1$: the residual dominates the reference; the abstract bound
  degrades to $> 1$ and provides no useful information.

### Sufficient $\\eta$ upper bound $\\bar{\\eta}_m$

When the transform has a known derivative supremum
$\\rho = \\sup_{y \\in [y_-, y_+]} |\\phi^{(m+1)}(y)|$ over the support $[y_-, y_+]$,
Taylor's theorem gives a computable upper bound:

$$\\bar{\\eta}_m \\leq
  \\frac{\\rho \\cdot \\sqrt{\\mathbb{E}[|Y - \\mu_Y|^{2m+2}]}}{(m+1)! \\cdot \\mathrm{sd}(V_m)}.$$

If $\\bar{\\eta}_m < 1$, the abstract bound is valid with $\\bar{\\eta}_m$ in
place of $\\eta_m$.  This requires a **provably correct support interval**
$[y_-, y_+]$ that contains the output range almost surely.  The notebook
provides this through `BENCHMARK_OUTPUT_BOUNDS`.

### Abstract projection bound

Given $\\eta_m < 1$ and a Sobol index subset $\\mathscr{C}$, the projection
error satisfies:

$$\\bigl|\\mathsf{PE}_{\\mathscr{C}}(Z) - \\mathsf{PE}_{\\mathscr{C}}(V_m)\\bigr|
  \\leq
  \\frac{2\\eta_m \\sqrt{p}\\,(1 + \\sqrt{p}) + \\eta_m^2(1 + p)}{(1 - \\eta_m)^2},$$

where $p = \\mathsf{PE}_{\\mathscr{C}}(V_m)$ is the Taylor reference's projection
value and $\\mathsf{PE}$ is the normalized-projection shorthand for ANOVA
component, closed, or total-effect Sobol indices.

*In the notebook outputs, the bound is computed for singleton and
total-effect projection classes and reported as `projection_bound_s1_*` and
`projection_bound_st_*` columns.*
"""

BOUNDS_LOCAL_AFFINE = """\
## Local-affine diagnostics ($m = 1$ case)

When $\\phi'(\\mu_Y) \\neq 0$ — the **nonzero-slope** case — the $m=1$ Taylor
reference is:

$$V_1 = \\phi'(\\mu_Y)\\,(Y - \\mu_Y).$$

$V_1$ is an affine function of $Y$, so $V_1$ has the *same* Sobol structure as
$Y$ itself (all indices are identical to those of $Y$).  This means the
abstract projection bound in the $m=1$ case also bounds the distance from
$\\phi(Y)$'s Sobol indices to those of $Y$.

### The diagnostic quantities $K$, $\\kappa$, $\\lambda$

The local-affine bound uses three benchmark- and transform-specific quantities:

**Moment ratio** $K$ (characterises the output distribution's heavy-tailedness):

$$K = \\frac{\\sqrt{\\mu_4}}{\\sigma_Y^2},$$

where $\\mu_4 = \\mathbb{E}[(Y - \\mu_Y)^4]$ is the fourth central moment and
$\\sigma_Y^2 = \\mathrm{Var}[Y]$.  Note: $K$ is estimated empirically from the
Saltelli sample.

**Nonlinearity ratio** $\\kappa$ (characterises how curved $\\phi$ is relative
to the output's scale):

$$\\kappa = \\frac{\\rho_2\\,\\sigma_Y}{|\\phi'(\\mu_Y)|},$$

where $\\rho_2 = \\sup_{y \\in [y_-, y_+]} |\\phi''(y)|$ is the second-derivative
supremum over the output support.

**Composite diagnostic** $\\lambda = K \\kappa$:

$$\\lambda = K \\kappa = \\frac{\\sqrt{\\mu_4}}{\\sigma_Y^2}
  \\cdot \\frac{\\rho_2\\,\\sigma_Y}{|\\phi'(\\mu_Y)|}
  = \\frac{\\rho_2\\sqrt{\\mu_4}}{\\sigma_Y\\,|\\phi'(\\mu_Y)|}.$$

The local-affine Corollary states that $\\eta_1 \\leq \\lambda/2$.  If
$\\lambda < 2$ (equivalently $\\eta_1 < 1$), the local-affine bound is finite:

$$\\bigl|\\mathsf{PE}_{\\mathscr{C}}(Z) - \\mathsf{PE}_{\\mathscr{C}}(Y)\\bigr|
  \\leq
  \\frac{\\lambda\\sqrt{p}\\,(1 + \\sqrt{p}) + \\tfrac14\\lambda^2(1 + p)}{(1 - \\lambda/2)^2}.$$

**Interpretation of $\\lambda$:**
- $\\lambda \\ll 1$: the transform is nearly affine over the output range;
  Sobol shifts are expected to be small.
- $\\lambda \\approx 1$: moderate nonlinearity; the bound is finite but loose.
- $\\lambda \\geq 2$: the sufficient stability condition fails; the local-affine
  bound may still hold with the empirical $\\eta_1$, but the analytic guarantee
  via $\\bar{\\eta}$ is unavailable.

### When the slope is zero (`zero_slope`)

If $\\phi'(\\mu_Y) = 0$ (a critical point), the local-affine reference $V_1 = 0$
is degenerate.  The $m=1$ bound is not meaningful; use $m=2$ (quadratic
reference) instead.  The `local_affine_status = "zero_slope"` flag marks this
case.
"""

BOUNDS_STATUS = """\
## Status vocabulary

Every row in `bounds_pair_status.csv` carries a `bounds_status` value that
records *why* the row reached the outcome it did.  Understanding these statuses
is important for correctly interpreting the coverage of the grid.

| Status | Meaning |
|--------|---------|
| `bounds_supported` | Caller-provided support bounds were used (from `BENCHMARK_OUTPUT_BOUNDS`).  Taylor-reference diagnostics were computed with a rigorous support interval.  **This is the strongest status.**  Note: the support interval itself may be analytically derived (e.g., Ishigami, Friedman) or empirically conservative (e.g., Borehole: lo=0 analytic, hi from N=1M + 5% buffer). |
| `bounds_diagnostic_sample_support` | Diagnostics were computed, but the support interval used was the empirical sample range (min/max of the Saltelli sample).  These are valid sample-range diagnostics but **not** theorem guarantees, because sample extremes may not contain the true support. |
| `bounds_not_scalar_output` | Benchmark produces multi-output (spatial/temporal) results; the bounds framework requires a scalar output. |
| `bounds_not_pointwise` | Transform is not pointwise (e.g., block average, COS); the bounds framework requires a pointwise function $\\phi$. |
| `bounds_not_smooth` | Transform is classified as nonsmooth (e.g., threshold, rectifier, absolute value); derivative metadata is not registered. |
| `bounds_no_derivative_metadata` | Transform is smooth and pointwise but lacks registered derivative information.  As of PR #11, this count is 0 for all catalog smooth+pointwise transforms. |
| `bounds_domain_invalid` | Benchmark output falls outside the transform's valid domain (e.g., log of a negative output). |
| `bounds_reference_zero_variance` | The Taylor reference $V_m$ has zero variance.  This occurs when all Taylor derivatives $\\phi^{(k)}(\\mu_Y)$ are near zero — typically because the transform is saturated or spiked and $\\mu_Y$ lies far outside the transform's active region.  The bound is degenerate; try a higher Taylor order or a better-centred transform. |
| `bounds_eta_ge_one` | Empirical $\\eta_m \\geq 1$; the residual dominates the reference and the bound provides no finite information. |
| `bounds_failed` | An unexpected runtime error occurred; inspect the `reason` column. |

### How `bounds_supported` works in practice

This notebook passes `benchmark_support=BENCHMARK_OUTPUT_BOUNDS` to
`evaluate_bounds_grid()`.  `BENCHMARK_OUTPUT_BOUNDS` contains entries for all
19 registered scalar benchmarks, so every smooth+pointwise pair with a scalar
benchmark will reach `bounds_supported` status.  Pairs with multi-output
benchmarks will reach `bounds_not_scalar_output`, and pairs with nonsmooth
or non-pointwise transforms will reach their respective statuses.

**Support provenance for `BENCHMARK_OUTPUT_BOUNDS`:**
- *Analytically exact* (12 benchmarks): Ishigami, SobolG, LinearModel,
  AdditiveQuadratic, CornerPeak, Friedman, MoonHerrera, DetPep8D, Rosenbrock,
  ProductPeak, PCETestFunction, CSTRReactor.
- *Empirically conservative, lower bound analytic* (5 benchmarks, all
  provably non-negative): Borehole, Piston, WingWeight, OTLCircuit,
  EnvironModel.  Upper bound from N=1M sample + 5% buffer.
- *Empirically conservative, both sides* (2 benchmarks): Morris, OakleyOHagan.
  Both bounds from N=1M sample + 5% buffer.

For rigorous theorem applications, use the analytically exact entries.
For the empirically conservative entries, treat results as strong diagnostics
but acknowledge the support interval is not provably tight.
"""

BOUNDS_CONFIG = """\
## Configuration

| Parameter | Env var | Default | Meaning |
|-----------|---------|---------|---------|
| `N_BASE` | `SABENCH_BOUNDS_N_BASE` | 128 | Base sample size for Saltelli design; actual evaluations = $N(d+2)$.  Use 512–2048 for publication-quality diagnostics. |
| `RNG_SEED` | `SABENCH_BOUNDS_SEED` | 20260429 | Random seed.  Fixed across all pairs. |
| `TAYLOR_ORDER` | `SABENCH_BOUNDS_TAYLOR_ORDER` | 2 | Order $m$ of the Taylor reference.  Order 1 = local-affine reference only.  Order 2 includes the quadratic term, which matters near critical points ($\\phi'(\\mu_Y) = 0$). |
| `MAX_BENCHMARKS` | `SABENCH_BOUNDS_MAX_BENCHMARKS` | 0 (all) | Truncate benchmark list to first $n$ entries.  0 = use all. |
| `MAX_TRANSFORMS` | `SABENCH_BOUNDS_MAX_TRANSFORMS` | 0 (all) | Truncate transform list to first $n$ entries.  0 = use all. |
| `OUTPUT_DIR` | `SABENCH_BOUNDS_OUTPUT_DIR` | `outputs/bounds_theorem_grid_analysis` | Directory for exported CSVs.  Created if absent. |

**Taylor order choice.**  Order 2 is appropriate for most purposes: it covers
the local-affine case ($m=1$ subcalculation) and the quadratic correction.
For exactly polynomial transforms of known degree, set the order to that
degree to get an exact (residual = 0) result.

**Transform ordering.** The configuration cell places smooth+pointwise
transforms first so that `MAX_TRANSFORMS` retains the most theoretically
interesting subset.

**Note on `bounds_supported` scope.** Only smooth+pointwise transforms with
scalar benchmarks produce `bounds_supported` rows.  The total grid includes
all registered transforms (smooth+pointwise ∪ nonsmooth ∪ non-pointwise);
non-applicable pairs are retained as status rows for completeness.
"""

BOUNDS_GRID_EXEC = """\
## Execute the theorem-assumption grid

The `evaluate_bounds_grid()` function processes each (benchmark, transform)
pair through the following pipeline:

1. **Applicability check** — verifies scalar output, pointwise mechanism,
   smooth tags, and derivative metadata.  Non-applicable pairs receive a
   descriptive status and no computation is attempted.

2. **Support resolution** — determines the support interval $[y_-, y_+]$:
   - If the benchmark key is in `BENCHMARK_OUTPUT_BOUNDS` (passed via
     `benchmark_support`): use the provided interval → `support_source = "provided_support"`.
   - Otherwise: use the sample range (min/max of the Saltelli output) →
     `support_source = "empirical_sample_range"`.

3. **Taylor reference computation** — evaluates $V_m$, $R_m$, and $\\eta_m$
   empirically; computes the sufficient $\\bar{\\eta}_m$ if derivative
   supremum is available.

4. **Projection bound computation** — evaluates the abstract bound formula
   for first-order and total-effect Sobol index subsets.

5. **Local-affine diagnostics** — computes $K$, $\\kappa$, $\\lambda$, and
   $\\eta_1$ for the $m=1$ case; reports `zero_slope` if $\\phi'(\\mu_Y) = 0$.

Pair-level failures (domain errors, numeric issues) are caught and recorded
as structured status rows rather than raising exceptions.

After the cell runs, `df_bounds` contains one row per pair.  The
`bounds_status` column records the outcome; numeric diagnostic columns are
populated only for applicable rows.
"""

BOUNDS_COLUMN_GLOSSARY = """\
## Exported tables and column glossary

### `bounds_pair_status.csv`

One row per (benchmark, transform) pair, regardless of applicability.

| Column | Description |
|--------|-------------|
| `benchmark_key` | Registry key |
| `transform_key` | Registry key |
| `bounds_status` | Primary status (see *Status vocabulary* above) |
| `reason` | Human-readable reason for non-`bounds_supported` rows |
| `benchmark_output_kind` | `scalar`, `spatial`, or `temporal` |
| `transform_mechanism` | `pointwise`, `spatial`, etc. |
| `transform_tags` | Semicolon-separated tags |
| `n_base`, `seed`, `n_inputs`, `n_evaluations` | Run parameters |
| `output_shape`, `output_finite`, `output_variance` | Output diagnostics |
| `taylor_order` | Taylor polynomial order used |

### `taylor_reference_results.csv`

Rows for all pairs that reached the Taylor-reference computation step
(i.e., `support_source` is not null).

| Column | Description |
|--------|-------------|
| `bounds_status` | `bounds_supported` or `bounds_diagnostic_sample_support` |
| `support_source` | `"provided_support"` (analytical/conservative bounds) or `"empirical_sample_range"` |
| `support_lower`, `support_upper` | The support interval $[y_-, y_+]$ used |
| `taylor_status` | `"computed"`, `"zero_reference_variance"`, or `"failed"` |
| `eta_empirical` | Empirical $\\eta_m = \\mathrm{sd}(R_m)/\\mathrm{sd}(V_m)$ |
| `eta_sufficient` | Sufficient upper bound $\\bar{\\eta}_m$ (or null if derivative supremum unavailable) |
| `eta_empirical_lt_one` | Boolean: whether $\\eta_m < 1$ (bound is finite) |
| `eta_sufficient_lt_one` | Boolean: whether $\\bar{\\eta}_m < 1$ |
| `projection_bound_s1_max` | Maximum first-order projection bound across inputs |
| `projection_bound_st_max` | Maximum total-effect projection bound across inputs |
| `projection_bound_s1_mean` | Mean first-order projection bound |
| `projection_bound_st_mean` | Mean total-effect projection bound |
| `reference_shift_s1_max` | Maximum first-order Sobol shift from $V_m$ to $Z$ |
| `reference_shift_st_max` | Maximum total-effect Sobol shift from $V_m$ to $Z$ |
| `reference_shift_s1_mean` | Mean first-order shift |
| `reference_shift_st_mean` | Mean total-effect shift |

### `local_affine_results.csv`

Rows for pairs that reached the local-affine diagnostic step.

| Column | Description |
|--------|-------------|
| `local_affine_status` | `"computed"` or `"zero_slope"` |
| `support_source` | Same as above |
| `sigma_y` | Standard deviation of benchmark output, $\\sigma_Y$ |
| `mu4` | Fourth central moment $\\mu_4 = \\mathbb{E}[(Y-\\mu_Y)^4]$ |
| `phi_prime_mu` | First derivative $\\phi'(\\mu_Y)$ |
| `rho2` | Second-derivative supremum $\\rho_2 = \\sup|\\phi''|$ over $[y_-, y_+]$ |
| `kappa` | Nonlinearity ratio $\\kappa = \\rho_2\\sigma_Y / |\\phi'(\\mu_Y)|$ |
| `moment_ratio` | Moment ratio $K = \\sqrt{\\mu_4}/\\sigma_Y^2$ |
| `lambda_value` | Composite diagnostic $\\lambda = K\\kappa$ |
| `eta_upper` | $\\eta_1 \\leq \\lambda/2$ (upper bound on first-order eta) |
| `lambda_lt_two` | Boolean: whether $\\lambda < 2$ (finite local-affine bound exists) |

### `bounds_summary.csv`

Two-column table (`bounds_status`, `count`) reporting how many rows of each
status appeared.  Ordered by the canonical status list for reproducibility.
"""

BOUNDS_INTERPRETATION = """\
## Interpreting results

### The fundamental distinction: theorem guarantee vs. sample diagnostic

**`bounds_supported` rows** use a pre-verified support interval
$[y_-, y_+]$ that contains the benchmark output almost surely.  For
analytically exact entries (e.g., Ishigami, Friedman), this is a *theorem
guarantee*: the sufficient $\\eta$ bound is computed with a provably correct
support, and the projection bound is a rigorous upper bound on the Sobol
index displacement.

**`bounds_diagnostic_sample_support` rows** use the observed sample range.
The sample extremes may underestimate the true support, so the sufficient
$\\eta$ bound is not guaranteed to be a valid upper bound — it is a diagnostic
estimate.  Do not report these as theorem guarantees.

### What the projection bound numbers mean

The projection bound columns give an upper bound on
$|\\mathrm{PE}_{\\mathscr{C}}(Z) - \\mathrm{PE}_{\\mathscr{C}}(V_m)|$, **not**
on $|S_u^Z - S_u^Y|$ directly (except in the $m=1$, nonzero-slope case where
$V_1$ has the same Sobol structure as $Y$).

- **When `projection_bound_s1_max` is small** (e.g., $< 0.05$): the
  transformed output's Sobol indices are close to the Taylor reference's
  indices.  For $m=1$, this also implies closeness to $Y$'s indices.
- **When the bound exceeds 1.0**: the bound is trivially satisfied but
  uninformative.  The actual shift may be much smaller.
- **When `eta_empirical_lt_one = False`**: the bound is not finite; the
  transform's nonlinearity over the output range is too large for the
  current Taylor order.

### What $\\lambda$ tells you

$\\lambda < 2$ is a *sufficient* condition for the local-affine bound to be
finite, not a necessary one.  A pair with $\\lambda > 2$ may still have a
finite empirical $\\eta_1 < 1$ and a useful bound.  The columns
`eta_empirical_lt_one` and `projection_bound_s1_max` give the bottom-line
diagnostic regardless of $\\lambda$.

Practically: $\\lambda \\ll 1$ means the transform behaves nearly affinely over
the output's typical range, and Sobol index changes will be small.
$\\lambda \\approx 1$ means moderate curvature is present.  $\\lambda > 2$ means
the curvature is large enough that the analytic guarantee breaks down; the
empirical $\\eta$ may still give useful information.

### The reference-shift columns

`reference_shift_*` columns measure the **empirical** Sobol shift from $V_m$
to $Z$, not the theoretical bound.  These are computed from the same Saltelli
sample as the Sobol estimates.  They provide a direct, sample-based view of
how much the transformation changed the profile relative to the Taylor
reference — complementing the bound columns.

### Sampling noise

As with the noncommutativity notebook, small values of `eta_empirical`,
`projection_bound_*`, or `reference_shift_*` at `N_BASE=128` carry sampling
uncertainty.  Use `N_BASE ≥ 512` for reliable diagnostics.

### Common patterns in the default grid output

When running with default settings, you will typically observe several
repeating patterns across the status summary.  These are not bugs — they
reflect the mathematical structure of the benchmark/transform landscape:

**`bounds_not_scalar_output`** is the largest group because the benchmark
registry includes spatial and temporal benchmarks (e.g., multi-output Gaussian
fields, Campbell 2D/3D).  The theorem applies only to scalar outputs.

**`bounds_not_pointwise`** covers spatial and temporal transforms (e.g.,
rank-normalisation, PCA projection, histogram equalisation).  The Taylor bound
requires that $\\phi$ acts pointwise on the scalar $Y$, so these are correctly
excluded.

**`bounds_not_smooth`** covers the 38+ non-smooth transforms in the registry
(absolute value, ReLU, sign, clipped, hard-threshold, etc.).  The Taylor bound
requires differentiability.

**`bounds_no_derivative_metadata`** covers transforms that are smooth and
pointwise but whose derivative functions are not registered.  These can be
diagnosed via `bounds_diagnostic_sample_support` by adding derivative metadata.

**`bounds_reference_zero_variance`** (~50 pairs in the default run) occurs when
a saturation or spike transform (e.g., `smooth_bump`, `erf_pointwise`,
`spike_gaussian`, `exp_neg_sq`, `logistic_pointwise`, `tanh_a10`) is applied to
a benchmark with a large output range and the benchmark mean $\\mu_Y$ falls far
outside the transform's active region.  In that case, all derivatives
$\\phi^{(k)}(\\mu_Y) \\approx 0$, so the Taylor reference $V_m \\approx 0$
identically — its variance is zero.  This is mathematically correct behaviour:
the transform is essentially saturated, and the Taylor expansion at $\\mu_Y$
carries no information.  Increasing the Taylor order or re-centering the
transform would be needed to make the bound informative.

**`bounds_eta_ge_one`** (~150 pairs) occurs when highly nonlinear transforms
(degree-4/5/6 polynomials, high-curvature tanhoids, $\\cos^2$, etc.) are
applied to benchmarks with wide output ranges.  The Taylor residual $R_m$
dominates the reference $V_m$, so $\\eta_m \\geq 1$ and the projection bound
diverges.  This is expected: the transform is too nonlinear over the sample
range for the current Taylor order to provide a finite bound.  Use a higher
Taylor order or a narrower transform/benchmark pair.

**`bounds_diagnostic_sample_support`** — diagnostics in this group were
computed successfully, but the support interval used was the empirical sample
range.  These are valid diagnostics for exploration, but they cannot be
reported as theorem guarantees.

**`bounds_supported`** — the strongest status.  Available for all 19 scalar
benchmarks in the default registry via `BENCHMARK_OUTPUT_BOUNDS`.  For
analytically exact entries (e.g., Ishigami $[-7.7, 7.7]$, Friedman
$[0, 30]$) the support is a true mathematical bound on $Y$, giving a rigorous
sufficient-$\\eta$ guarantee.  For empirically conservative entries (e.g.,
Borehole, Piston) the hi-side was set from $N = 10^6$ samples plus a 5%
buffer; these are practically conservative but not strictly theorem-guaranteed.
"""


def build_noncomm_nb(original_nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    code_cells = preserve_code(original_nb)
    assert len(code_cells) == 5, f"Expected 5 code cells, got {len(code_cells)}"
    c = code_cells

    nb = nbformat.v4.new_notebook()
    nb.metadata = original_nb.metadata
    nb.cells = [
        md(NONCOMM_INTRO),
        md(NONCOMM_BACKGROUND_SOBOL),
        md(NONCOMM_THEOREM),
        md(NONCOMM_METRICS),
        md(NONCOMM_CONFIG),
        c[0],  # configuration code
        md(NONCOMM_GRID_EXEC),
        c[1],  # grid execution code
        md(NONCOMM_COLUMN_GLOSSARY),
        c[2],  # export code
        md(NONCOMM_SUMMARIES),
        c[3],  # summary code
        md(NONCOMM_INTERPRETATION),
        c[4],  # quick checks code
    ]
    return nb


def build_bounds_nb(original_nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    code_cells = preserve_code(original_nb)
    assert len(code_cells) == 3, f"Expected 3 code cells, got {len(code_cells)}"
    c = code_cells

    nb = nbformat.v4.new_notebook()
    nb.metadata = original_nb.metadata
    nb.cells = [
        md(BOUNDS_INTRO),
        md(BOUNDS_BACKGROUND),
        md(BOUNDS_TAYLOR),
        md(BOUNDS_LOCAL_AFFINE),
        md(BOUNDS_STATUS),
        md(BOUNDS_CONFIG),
        c[0],  # configuration code
        md(BOUNDS_GRID_EXEC),
        c[1],  # grid execution code
        md(BOUNDS_COLUMN_GLOSSARY),
        c[2],  # export + summary display code
        md(BOUNDS_INTERPRETATION),
    ]
    return nb


def strip_outputs(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Remove all cell outputs and reset execution counts."""
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return nb


def main() -> None:
    repo = Path(__file__).parent.parent
    nb_dir = repo / "notebooks"

    for name, builder in [
        ("noncommutativity_grid_analysis.ipynb", build_noncomm_nb),
        ("bounds_theorem_grid_analysis.ipynb", build_bounds_nb),
    ]:
        path = nb_dir / name
        with open(path) as f:
            orig = nbformat.read(f, as_version=4)

        # Verify code cells are unchanged by checking source strings
        orig_code_sources = [c.source for c in orig.cells if c.cell_type == "code"]

        new_nb = builder(orig)
        new_nb = strip_outputs(new_nb)

        # Invariant: code cells must be byte-for-byte identical
        new_code_sources = [c.source for c in new_nb.cells if c.cell_type == "code"]
        assert orig_code_sources == new_code_sources, (
            f"Code cells changed in {name}!\n"
            f"Original count: {len(orig_code_sources)}\n"
            f"New count: {len(new_code_sources)}"
        )

        with open(path, "w") as f:
            nbformat.write(new_nb, f)
        print(
            f"Rewrote {name}: "
            f"{len([c for c in new_nb.cells if c.cell_type == 'markdown'])} markdown cells, "
            f"{len([c for c in new_nb.cells if c.cell_type == 'code'])} code cells"
        )


if __name__ == "__main__":
    main()
