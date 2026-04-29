# Engineering Manifest

Last updated: 2026-04-29

This manifest is the planning document for release-readiness work in this
repository. Treat the live repository and tests as authoritative; treat
`docs/MEMORY.md` and older handoff notes as advisory continuity context only.

## Current Repository State

- `sabench` is a Python 3.10+ package for benchmark-driven Sobol sensitivity
  analysis and output-transformation studies.
- The live checkout uses Pixi for local and CI validation.
- `test_repo.sh` is the authoritative local gate and delegates core checks to
  named Pixi tasks.
- The current release line is `0.3.0`.
- Tests live outside the runtime package under `tests/`.
- Notebooks must remain clean: no committed execution counts or outputs.

## Non-Negotiable Constraints

- Treat the live repository on disk as the only source of truth.
- Work in small atomic slices on task branches.
- Use test-first development for implementation changes.
- Do not add compatibility shims unless explicitly requested.
- Do not weaken, skip, xfail, or narrow tests to make validation pass.
- Do not commit generated outputs, caches, build products, local archives, or
  platform artifacts.
- Metadata snapshots must be generated from, or validated against, canonical
  registry definitions.
- Notebook scientific logic must live in tested reusable analysis utilities, not
  only in notebook cells.

## Validation Gate

Authoritative local gate:

```bash
./test_repo.sh --check-only
```

Minimum slice validation:

```bash
python -m compileall -q sabench tests scripts
bash -n test_repo.sh
PYTHONPATH=. pytest -q tests/analysis --tb=short
PYTHONPATH=. pytest -q tests/integration --tb=short
```

Notebook-slice validation:

```bash
PYTHONPATH=. pytest -q tests/integration/test_noncommutativity_grid_notebook.py --tb=short
PYTHONPATH=. pytest -q tests/integration/test_bounds_theorem_grid_notebook.py --tb=short
./test_repo.sh --check-only
```

Final release validation:

```bash
./test_repo.sh --clean
./test_repo.sh --check-only
git status --short
gh pr status --repo dhetting/sabench
```

## Priority Order

### P0 — Safety / Correctness / Broken Gate

- [ ] Keep the local gate and CI aligned around the same Pixi tasks.
- [ ] Ensure generated notebook output directories are ignored, cleaned, and
  excluded from source bundles before final release.
- [ ] Keep notebooks output-free and execution-count-free.
- [ ] Run the full local gate before final release handoff.

### P1 — Required Feature Completion

- [x] Phase A release baseline verified where covered by live files and tests.
- [x] Phase B empirical noncommutativity utilities and notebook are present.
- [ ] Phase C bounds-theorem grid engine and notebook remain incomplete.
- [ ] Phase D final release-readiness pass remains incomplete.

### P2 — Hardening

- [ ] Add bounds-grid tests that distinguish theorem-supported calculations,
  empirical sample-support diagnostics, non-applicable pairs, and failed pairs.
- [ ] Add notebook-contract tests for the bounds-theorem notebook.
- [ ] Document both analysis notebooks in release-facing docs after they pass
  their contract tests.

### P3 — Documentation And Examples

- [ ] Update `docs/MEMORY.md` after completing the notebook and finalization
  slices.
- [ ] Document final release-readiness status.
- [ ] Resolve release-paper citation placeholders when external identifiers are
  available.

## Phase A — Current Release Baseline

Verified against the live repo on 2026-04-29 unless otherwise noted.

- [x] Transform monolith removed.
  - Verified by absence of `sabench/transforms/transforms.py` and tests in
    `tests/transforms/test_transform_registry_contract.py`.
- [x] Typed benchmark and transform registries are canonical.
  - Verified by `sabench/benchmarks/registry.py`,
    `sabench/transforms/registry.py`, and registry tests.
- [x] Metadata snapshots are generated or validated from registry definitions.
  - Verified by `tests/integration/test_registry_metadata_exports.py`.
- [x] Source-bundle hygiene hardened.
  - Verified by `scripts/build_source_bundle.py` and
    `tests/integration/test_source_bundle_hygiene.py`.
- [x] Recursive clean hardened.
  - Verified by `test_repo.sh` clean-stage logic and
    `tests/integration/test_gate_contract_alignment.py`.
- [x] Tracked generated-artifact guardrails added.
  - Verified by `.gitignore` and
    `tests/integration/test_gate_contract_alignment.py`.
- [x] Package-smoke testing added.
  - Verified by `scripts/check_built_distribution.py`, `pixi.toml`,
    `test_repo.sh`, and `.github/workflows/ci.yml`.
- [x] README/demo API consistency tested.
  - Verified by `tests/integration/test_root_api_narrow.py`.

## Phase B — Empirical Noncommutativity Notebook

Required track for `notebooks/02_noncommutativity_grid_analysis.ipynb`, which
computes empirical metrics from `noncommutativity_detailed.tex`.

- [x] Verify or implement `sabench.analysis.noncommutativity`.
  - Present and covered by `tests/analysis/test_noncommutativity.py`.
- [x] Verify or implement `sabench.analysis.grid`.
  - Present and covered by `tests/analysis/test_noncommutativity_grid.py`.
- [x] Verify or implement `notebooks/02_noncommutativity_grid_analysis.ipynb`.
  - Present, registry-driven, and output-clean.
- [x] Verify notebook-contract tests.
  - Present at `tests/integration/test_noncommutativity_grid_notebook.py`.
- [x] Verify exported tables.
  - Contract test verifies `pair_status.csv`,
    `noncommutativity_metrics.csv`, `summary_by_transform.csv`, and
    `summary_by_benchmark.csv`.
- [x] Verify primary metrics.
  - Covered metrics include `D_s1`, `D_st`, `delta_s1`, `delta_st`,
    threshold flips, top-k changes, max/mean absolute shifts, Spearman rank
    correlation, and top-driver changes.

## Phase C — Bounds-Theorem Notebook

Required track for `notebooks/03_bounds_theorem_grid_analysis.ipynb`, which will
compute theorem-oriented diagnostics from `bounds_memo_v22.tex`.

- [x] Verify or implement `sabench.analysis.bounds`.
  - Present and covered by `tests/analysis/test_bounds.py`.
- [ ] Add or verify a bounds grid engine.
  - Expected path: `sabench/analysis/bounds_grid.py`.
- [ ] Add or verify tests for bounds-grid classification and calculations.
  - Expected path: `tests/analysis/test_bounds_grid.py`.
- [ ] Implement `notebooks/03_bounds_theorem_grid_analysis.ipynb`.
- [ ] Add notebook-contract tests.
  - Expected path: `tests/integration/test_bounds_theorem_grid_notebook.py`.
- [ ] Verify exported tables:
  - `bounds_pair_status.csv`
  - `taylor_reference_results.csv`
  - `local_affine_results.csv`
  - `bounds_summary.csv`
- [ ] Ensure theorem statuses distinguish:
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
- [ ] Ensure notebooks and exported tables clearly state when a theorem-bound
  comparison is against the Taylor reference `V_m`, not directly against `Y`.

## Phase D — Finalization

- [ ] Notebook execution gate passes.
- [ ] Full local gate passes.
- [ ] Package smoke passes.
- [ ] GitHub CI passes.
- [ ] `docs/MEMORY.md` updated.
- [ ] Final release-readiness status documented.

## Completed Work

- Transform implementation split into focused modules.
- Canonical transform catalog and typed registries established.
- Registry metadata export validation added.
- Source-bundle and generated-artifact hygiene hardened.
- Built-wheel package smoke test added.
- Root API narrowed and README/demo consistency guarded.
- Empirical noncommutativity metric utilities, grid engine, notebook, and
  notebook-contract tests added.
- Taylor-reference and local-affine bound utility functions added.

## Known Risks

- Bounds-theorem grid execution is not yet implemented.
- Bounds notebook and contract tests are not yet implemented.
- `test_repo.sh` performs environment sync and build/package smoke work, so full
  validation can be slower than targeted test commands.
- Generated notebook output directory handling should be reviewed before final
  release.
- README citation contains a JOSS DOI placeholder pending external publication
  metadata.

## Deferred Work

- Broaden derivative metadata beyond the initial smooth pointwise transform set
  only when needed and tested.
- Add release-paper or README updates after both notebook tracks are complete.
- Final publish/release workflow remains out of scope until full gate and CI
  pass.
