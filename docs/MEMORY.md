# MEMORY — sabench release-readiness continuation

Last updated: 2026-04-30

This file is continuity context only. It is advisory, not authoritative. In any
new chat or future slice, audit the live repo on disk before trusting this file,
previous bundles, prior assistant claims, GitHub PR state, or claimed local
state.

## Current Working Posture

The `sabench` repo is in late-stage release-readiness hardening. The transform
monolith is retired in the live stack: transform implementations live in focused
modules, canonical transform catalog data lives in `sabench/transforms/catalog.py`,
and transform evaluation helpers live in `sabench/transforms/evaluation.py`.

All implementation work through Phase C is merged into `main`. The PR stack
(docs/engineering-manifest, fix/notebook-output-hygiene, feat/bounds-grid-engine,
docs/bounds-theorem-notebook, docs/notebook-docs-memory) has been squash-merged.
The current state of `main` as of 2026-04-30:

- `HEAD` = `aaa4b93 feat: add bounds grid and notebook finalization`
- No open PRs
- No stale branches (only `main` and legacy `master`)
- Both analysis notebooks present, clean, and tested

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
- `notebooks/02_noncommutativity_grid_analysis.ipynb`
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
- `notebooks/03_bounds_theorem_grid_analysis.ipynb`
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

Bounds statuses:

- `bounds_supported`
- `bounds_diagnostic_sample_support`
- `bounds_not_scalar_output`
- `bounds_not_pointwise`
- `bounds_not_smooth`
- `bounds_no_derivative_metadata`
- `bounds_domain_invalid`
- `bounds_reference_zero_variance`
- `bounds_eta_ge_one`
- `bounds_failed`

Status: **complete** — all files present, tested, and notebook-contract verified.

## Validation Commands To Prefer

Use the repo-defined gate if it differs, but start with:

```bash
cd ~/src/sabench
python -m compileall -q sabench tests scripts
bash -n test_repo.sh
PYTHONPATH=. pytest -q tests/analysis --tb=short
PYTHONPATH=. pytest -q tests/integration --tb=short
./test_repo.sh --check-only --no-notebook
```

Full local gate (may be slow due to environment sync and build):

```bash
./test_repo.sh --check-only
```

For final release:

```bash
./test_repo.sh --clean
./test_repo.sh --check-only
git status --short
gh pr status --repo dhetting/sabench
```

## Remaining Work

1. Run the notebook execution gate end-to-end (both notebooks).
2. Run the full local gate (`./test_repo.sh --check-only`) with clean state.
3. Confirm GitHub CI passes on main.
4. Document final release-readiness status in `docs/ENGINEERING_MANIFEST.md`
   Phase D section.
5. Only then proceed to release tagging/publishing workflow if explicitly
   requested.

## Handoff Notes

- Do not blindly stage generated artifacts. Validation may produce ignored
  `.coverage`, `coverage.xml`, `.pixi/`, cache directories, and `dist/`.
- Keep notebooks clean: no committed outputs and no execution counts.
- Outputs from notebook execution land under `outputs/` which is gitignored.
- The `docs/manuscripts/` directory contains local LaTeX source files and is
  tracked in git as scientific reference context.
