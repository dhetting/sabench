# Engineering Manifest

Last updated: 2026-04-30

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
- [x] Phase B empirical noncommutativity utilities and notebook are present and
  tested.
- [x] Phase C bounds-theorem grid engine and notebook are present and tested.
- [ ] Phase D final release-readiness pass: notebook execution gate, full gate,
  and `docs/MEMORY.md` refresh remain.

### P2 — Hardening

- [x] Bounds-grid tests distinguish theorem-supported calculations, empirical
  sample-support diagnostics, non-applicable pairs, and failed pairs.
- [x] Notebook-contract tests for both notebooks are present.
- [x] Both analysis notebooks documented in README.

### P3 — Documentation And Examples

- [x] `docs/MEMORY.md` updated to reflect live state.
- [ ] Final release-readiness status documented after full gate passes.
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

Required track for `notebooks/03_bounds_theorem_grid_analysis.ipynb`, which
computes theorem-oriented diagnostics from `bounds_memo_v22.tex`.

- [x] Verify or implement `sabench.analysis.bounds`.
  - Present and covered by `tests/analysis/test_bounds.py`.
- [x] Add or verify a bounds grid engine.
  - Present at `sabench/analysis/bounds_grid.py`.
- [x] Add or verify tests for bounds-grid classification and calculations.
  - Present at `tests/analysis/test_bounds_grid.py`.
- [x] Implement `notebooks/03_bounds_theorem_grid_analysis.ipynb`.
  - Present, registry-driven, and output-clean.
- [x] Add notebook-contract tests.
  - Present at `tests/integration/test_bounds_theorem_grid_notebook.py`.
- [x] Verify exported tables.
  - Contract test verifies `bounds_pair_status.csv`,
    `taylor_reference_results.csv`, `local_affine_results.csv`, and
    `bounds_summary.csv`.
- [x] Ensure theorem statuses distinguish:
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
- [x] Ensure notebooks and exported tables clearly state when a theorem-bound
  comparison is against the Taylor reference `V_m`, not directly against `Y`.

  Taylor reference: `V_m = sum_{k=1}^m phi^(k)(mu_Y) (Y - mu_Y)^k / k!`
  Residual: `R_m = phi(Y) - phi(mu_Y) - V_m`
  Empirical eta: `eta_m = sd(R_m) / sd(V_m)`
  Abstract projection bound: `[2 eta sqrt(p)(1+sqrt(p)) + eta^2(1+p)] / (1-eta)^2`
  Local-affine diagnostics: `lambda = K kappa`,
  `K = sqrt(mu4) / sigma_Y^2`, `kappa = rho2 sigma_Y / |phi'(mu_Y)|`

## Phase D — Finalization

- [ ] Notebook execution gate passes (run both notebooks end-to-end).
- [ ] Full local gate passes: `./test_repo.sh --check-only`.
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
- Bounds grid engine, classification statuses, and notebook-contract tests added.
- Both analysis notebooks (noncommutativity and bounds-theorem) documented in README.
- Engineering manifest and memory files updated to reflect live state.

## Known Risks

- Full notebook execution gate (running both notebooks end-to-end) has not yet
  been run in CI; currently verified only through notebook-contract tests.
- README citation contains a JOSS DOI placeholder pending external publication
  metadata.
- `test_repo.sh` performs environment sync and build/package smoke work, so full
  validation can be slower than targeted test commands.

## Deferred Work

- Broaden derivative metadata beyond the initial smooth pointwise transform set
  only when needed and tested.
- Final publish/release workflow remains out of scope until full gate and CI pass.
