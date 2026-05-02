# MEMORY — sabench release-readiness continuation

Last updated: 2026-05-02 (post-PR #18)

This file is continuity context only. It is advisory, not authoritative. In any
new chat or future slice, audit the live repo on disk before trusting this file,
previous bundles, prior assistant claims, GitHub PR state, or claimed local
state.

## Current Working Posture

The `sabench` repo has completed all Phase A–D release-readiness work. The
transform monolith is retired; typed registries are canonical; both analysis
notebooks execute cleanly; the full local gate and GitHub CI pass.

Current state of `main` as of 2026-05-02 (HEAD `29320f4`, post-PR #18):

- All Phase A–D work merged and verified.
- No open PRs, no stale branches, clean worktree.
- Local gate (`./test_repo.sh --check-only --no-notebook`) passes all stages.
- GitHub CI passes on `main`.
- Both analysis notebooks execute end-to-end and export all expected CSVs.
- Pre-commit mypy hook is fast (uses pixi-managed mypy, no isolated virtualenv).
- Transform catalog corrected: 5 transforms moved from smooth to nonsmooth
  (legendre_p3, chebyshev_t4, damped_sin, donut, signed_power_p15). PR #10.
  SMOOTH_TRANSFORMS=85, NONSMOOTH_TRANSFORMS=32.
- Derivative metadata registered for all 36 smooth+pointwise transforms in
  `_SMOOTH_POINTWISE_ANALYSES` (38 entries total). PR #11.
  `bounds_no_derivative_metadata` is 0 for all catalog-registered smooth+pointwise
  pairs.
- `BENCHMARK_OUTPUT_BOUNDS` extended to all 19 registered scalar benchmarks
  (analytically exact for 12, empirically conservative N=1M+5% buffer for 7).
  `benchmark_support` parameter added to `evaluate_bounds_grid()`. The bounds
  notebook passes these bounds, promoting qualifying pairs to `bounds_supported`.
  PRs #13, #15.
- Both analysis notebooks comprehensively documented (verbose markdown,
  background, formulas, column glossaries, interpretation). PR #17.
- `pair_status` column glossary corrected (`included`/`excluded` not
  `computed`/`skipped_incompatible`); all `metrics_status` values documented.
  Overflow warnings from exp transforms suppressed with `np.errstate` at
  per-pair evaluation sites; structured status rows capture all exceptional
  cases. Bounds interpretation section added explaining zero-reference-variance
  and eta≥1 patterns. PR #18.

## Non-Negotiable Operating Rules

- Treat the live repo as the only source of truth.
- Audit first, then patch.
- Strict test-driven development for implementation changes.
- Atomic slices only.
- No backward-compatibility shims unless explicitly requested.
- No hacks or workarounds.
- Fix root causes.
- Tests belong outside the runtime package.
- Metadata must be generated from, or validated against, canonical code
  definitions.
- Do not claim validation that was not actually run.
- Use Pixi-backed repo gates when available.

## Current Notebook Tracks

### Empirical Noncommutativity Grid

Files:

- `sabench/analysis/noncommutativity.py`
- `sabench/analysis/grid.py`
- `notebooks/noncommutativity_grid_analysis.ipynb`
- `tests/analysis/test_noncommutativity.py`
- `tests/analysis/test_noncommutativity_grid.py`
- `tests/integration/test_noncommutativity_grid_notebook.py`

Purpose:

- Compute registry-driven empirical benchmark × transform non-commutativity
  metrics from `noncommutativity_detailed.tex`.
- Export `pair_status.csv`, `noncommutativity_metrics.csv`,
  `summary_by_transform.csv`, and `summary_by_benchmark.csv` to
  `outputs/noncommutativity_grid_analysis/`.
- Keep notebook scientific logic in tested reusable analysis utilities.

Key implementation notes:

- `pair_status` column values: `"included"` (computation attempted) or
  `"excluded"` (incompatible transform/benchmark output kind).
- `metrics_status` values: `"computed"`, `"not_applicable"`,
  `"failed_raw_evaluation"`, `"failed_raw_output_validation"`,
  `"failed_transform_evaluation"`, `"failed_transformed_output_validation"`,
  `"failed_sobol_estimation"`, `"failed_metric_computation"`.
- `failed_metric_computation` is expected for transforms like `exp_pos_sq` on
  large-output benchmarks: Y is technically finite (~1e304) but squaring for
  Jansen variance estimation overflows to inf. All captured as structured rows.
- Overflow warnings are suppressed via `np.errstate(over='ignore',
  invalid='ignore')` in `evaluate_one_pair()`.

Primary metrics:

- `D_s1`, `D_st`
- `delta_s1`, `delta_st`
- threshold flips
- top-k changes
- max/mean absolute shifts
- Spearman rank correlation
- top-driver changes

Status: **complete** — all files present, tested, and notebook-contract verified.

### Bounds-Theorem Grid

Files:

- `sabench/analysis/bounds.py`
- `sabench/analysis/bounds_grid.py`
- `notebooks/bounds_theorem_grid_analysis.ipynb`
- `tests/analysis/test_bounds.py`
- `tests/analysis/test_bounds_grid.py`
- `tests/integration/test_bounds_theorem_grid_notebook.py`

Purpose:

- Compute theorem-oriented Taylor-reference and local-affine diagnostics from
  `bounds_memo_v22.tex`.
- Export `bounds_pair_status.csv`, `taylor_reference_results.csv`,
  `local_affine_results.csv`, and `bounds_summary.csv` to
  `outputs/bounds_theorem_grid_analysis/`.
- Clearly separate theorem-supported rows from empirical sample-range
  diagnostics, non-applicable rows, and failed rows.
- State that projection-bound comparisons are against the Taylor reference
  `V_m`, not directly against original output `Y`.

Bounds statuses (`sabench/analysis/bounds.py`):

- `bounds_supported` — explicit theoretical support provided via
  `BENCHMARK_OUTPUT_BOUNDS`; all 19 scalar benchmarks covered (12 analytically
  exact, 7 empirically conservative N=1M+5% buffer).
- `bounds_diagnostic_sample_support` — smooth+pointwise with derivative
  metadata; empirical sample-range support used
- `bounds_not_scalar_output`
- `bounds_not_pointwise`
- `bounds_not_smooth`
- `bounds_no_derivative_metadata` — currently 0 for all catalog pairs
- `bounds_domain_invalid`
- `bounds_reference_zero_variance` — occurs when saturation/spike transforms
  have all derivatives ≈0 at μ_Y (benchmark mean far from transform's active
  region); mathematically correct, ~50 pairs in default run
- `bounds_eta_ge_one` — highly nonlinear transforms; residual dominates
  reference; ~150 pairs in default run; bound uninformative
- `bounds_failed`

Overflow warnings from exp-derivative evaluation are suppressed via
`np.errstate(over='ignore', invalid='ignore')` in `evaluate_bounds_pair()`.

Analytical output bounds (`BENCHMARK_OUTPUT_BOUNDS`) — all 19 scalar benchmarks:
- Ishigami (a=7, b=0.1): Y∈[-(1+b·π⁴), 1+a+b·π⁴]
- SobolG (8 params): Y∈[0, ∏(2+aᵢ)/(1+aᵢ)]
- LinearModel: Y∈[0, 6.6]
- AdditiveQuadratic: Y∈[0, 10.5]
- CornerPeak: Y∈[(1+Σc)⁻⁷, 1]
- Friedman: Y∈[0, 30]
- MoonHerrera: Y∈[1, exp(c·Σwᵢ)]
- DetPep8D: Y∈[0, 70+16√2]
- Rosenbrock (d=4): Y∈[0, 10827]
- ProductPeak: Y∈[∏1/(cᵢ²+0.25), ∏1/cᵢ²]
- PCETestFunction: Y∈[-1.5−3.8²/(4·4.8), 9.5]
- CSTRReactor: Y∈[0, 0.8]
- Borehole, Piston, WingWeight, OTLCircuit, EnvironModel: lo=0 analytical, hi empirical
- Morris, OakleyOHagan: both sides empirical (N=1M, 5% buffer)

Status: **complete** — all files present, tested, notebook-contract verified,
and end-to-end execution validated (exports all 4 expected CSVs).

## Release Status

All Phases A–D are complete as of 2026-05-02:

- Local gate passes (`./test_repo.sh --check-only --no-notebook` all green,
  including fast pre-commit mypy via pixi-delegating hook).
- GitHub CI passes on `main` at HEAD `29320f4`.
- Both analysis notebooks execute end-to-end, exporting all expected CSVs.
- Pre-commit hooks fast and stable.
- 124 tests passing.

The repo is release-ready. Remaining deferred items:
- Resolving the JOSS DOI placeholder (external dependency).
- Release tagging and PyPI publish when explicitly requested.

## Validation Commands To Prefer

```bash
cd ~/src/sabench
python -m compileall -q sabench tests scripts
bash -n test_repo.sh
PYTHONPATH=. pytest -q tests/analysis --tb=short
PYTHONPATH=. pytest -q tests/integration --tb=short
./test_repo.sh --check-only --no-notebook
```

## Handoff Notes

- Do not blindly stage generated artifacts. Validation may produce ignored
  `.coverage`, `coverage.xml`, `.pixi/`, cache directories, and `dist/`.
- Keep notebooks clean: no committed outputs and no execution counts.
- Outputs from notebook execution land under `outputs/<notebook_name>/`
  relative to the repo root (gitignored).
- The `docs/manuscripts/` directory contains LaTeX source files tracked in git
  as scientific reference context.
