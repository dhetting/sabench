# MEMORY — sabench release-readiness continuation

Last updated: 2026-05-01 (post-PR #11)

This file is continuity context only. It is advisory, not authoritative. In any
new chat or future slice, audit the live repo on disk before trusting this file,
previous bundles, prior assistant claims, GitHub PR state, or claimed local
state.

## Current Working Posture

The `sabench` repo has completed all Phase A–D release-readiness work. The
transform monolith is retired; typed registries are canonical; both analysis
notebooks execute cleanly; the full local gate and GitHub CI pass.

Current state of `main` as of 2026-05-01 (HEAD `ca3c7e1`, post-PR #11):

- All Phase A–D work merged and verified.
- No open PRs, no stale branches.
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
  pairs. Pairs instead report `bounds_diagnostic_sample_support`.
- `bounds_supported` is 0 for all pairs; requires explicit per-benchmark
  theoretical output support bounds passed via `support_by_pair` to
  `evaluate_bounds_grid`. This is a known deferred item.

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
  `summary_by_transform.csv`, and `summary_by_benchmark.csv`.
- Keep notebook scientific logic in tested reusable analysis utilities.

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
  `local_affine_results.csv`, and `bounds_summary.csv`.
- Clearly separate theorem-supported rows from empirical sample-range
  diagnostics, non-applicable rows, and failed rows.
- State that projection-bound comparisons are against the Taylor reference
  `V_m`, not directly against original output `Y`.

Bounds statuses (`sabench/analysis/bounds.py`):

- `bounds_supported` — explicit theoretical support provided; currently 0 pairs
  in all-catalog grid (deferred: per-benchmark support bounds not yet registered)
- `bounds_diagnostic_sample_support` — smooth+pointwise with derivative metadata;
  all 36 smooth+pointwise catalog transforms now reach this status
- `bounds_not_scalar_output`
- `bounds_not_pointwise`
- `bounds_not_smooth`
- `bounds_no_derivative_metadata` — currently 0 for all catalog pairs (all
  smooth+pointwise transforms have registered derivative metadata as of PR #11)
- `bounds_domain_invalid`
- `bounds_reference_zero_variance`
- `bounds_eta_ge_one`
- `bounds_failed`

Status: **complete** — all files present, tested, notebook-contract verified,
and end-to-end execution validated (exports all 4 expected CSVs).

## Release Status

All Phases A–D are complete as of 2026-05-01:

- Local gate passes (`./test_repo.sh --check-only --no-notebook` all green,
  including fast pre-commit mypy via pixi-delegating hook).
- GitHub CI passes on `main`.
- Both analysis notebooks execute end-to-end in fast mode, exporting all
  expected CSVs.
- Pre-commit hooks fast and stable.

The repo is release-ready. Remaining deferred items:
- Adding per-benchmark theoretical output support bounds to enable
  `bounds_supported` pairs in the bounds grid (currently all pairs report
  `bounds_diagnostic_sample_support`).
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
- Outputs from notebook execution land under `notebooks/outputs/` (gitignored).
- The `docs/manuscripts/` directory contains LaTeX source files tracked in git
  as scientific reference context.
