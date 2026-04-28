# MEMORY — sabench refactor and finalization handoff

Updated: 2026-04-28

## Purpose of this file

Use this file as continuity context only. It is advisory, not authoritative. In every new chat, inspect the actual repo copy provided by the user and treat that live copy as the only source of truth.

## User goals

The user is finalizing the `sabench` repo for a clean, publication-quality architecture centered on two primary scientific concepts:

- `benchmarks`
- `transforms`

The user wants strict engineering discipline:

- audit the actual live repo first
- test-first development
- atomic slices only
- no backward-compatibility shims
- no hacks or workarounds
- fix root causes rather than symptoms
- tests outside the runtime package
- canonical typed registries
- metadata generated from, or validated against, canonical code definitions
- source/changed-file bundles with repo-relative paths at zip root
- never package `.git`, caches, coverage artifacts, old zips, macOS metadata, or transient files

## Non-negotiable workflow requirements

At the start of any continuation chat:

1. Inspect the actual repo copy provided in that chat.
2. Determine the real Git/worktree state.
3. Do not trust this memory file, prior bundles, or summaries as authoritative.
4. Re-audit the current architecture before changing files.
5. Produce or update a prioritized engineering manifest before implementing if the task is broad.
6. Implement exactly one atomic slice unless the user explicitly changes scope.
7. Write failing tests first for the chosen slice.
8. Run compile/syntax checks, targeted tests, full pytest, and the full local gate as far as the environment allows.
9. Provide a changed-files bundle plus exact local apply/test/commit commands.

## Latest known repo source and caveat

The latest live repo archive used in the most recent chat was:

- `sabench-v0.5.zip`

That archive was treated as authoritative only for that chat. It contained stale/transient artifacts such as `.git`, `__MACOSX`, `.coverage`, `coverage.xml`, and `diff.txt`, which motivated later source-bundle hygiene hardening.

Several cumulative changed-file bundles were produced after that archive. They should not be assumed to match the user's local repo unless the user confirms they were applied and validated locally.

## High-level architecture status after latest produced slices

### Benchmarks

Earlier refactor status, still advisory until re-audited:

- `sabench/benchmarks/` is the canonical benchmark package.
- Benchmark families were moved under:
  - `sabench/benchmarks/scalar/`
  - `sabench/benchmarks/spatial/`
  - `sabench/benchmarks/functional/`
- `sabench/_base.py` was removed.
- `sabench/benchmarks/base.py` is canonical.
- Typed benchmark registry exists in `sabench/benchmarks/registry.py`.
- Typed benchmark metadata/spec exists in `sabench/benchmarks/types.py`.
- Canonical benchmark metadata export exists at:
  - `sabench/metadata/benchmarks_registry_metadata.json`

### Transforms before the latest chat

At the start of the latest audited work, the transform side was partially modularized but still depended heavily on `sabench/transforms/transforms.py`.

The audit found:

- `tests/integration/test_ci_workflow_alignment.py` was already absent.
- Canonical registry metadata snapshots matched generated exports before implementation.
- `sabench/transforms/transforms.py` still defined 23 transform functions.
- 24 transform registry entries still pointed to `sabench.transforms.transforms`.

### Transform monolith endgame completed in generated slices

The following cumulative slices were produced after `sabench-v0.5.zip`.

#### Slice: spatial aggregation / field-op extraction

Moved out of `sabench/transforms/transforms.py`:

- Into `sabench/transforms/aggregation.py`:
  - `t_regional_mean`
  - `_block_avg`
  - `t_block_2x2`
  - `t_block_4x4`
  - `t_block_8x8`
  - `t_exceedance_area`
- Into `sabench/transforms/field_ops.py`:
  - `t_matern_smooth`
  - `t_laplacian_roughness`
  - `t_contour_exceedance`
  - `t_isoline_length`

Updated tests and metadata:

- `tests/transforms/test_transform_module_split.py`
- `tests/transforms/test_transform_registry.py`
- `tests/integration/test_registry_metadata_exports.py`
- `sabench/metadata/transforms_registry_metadata.json`

#### Slice: temporal operator extraction

Moved out of `sabench/transforms/transforms.py`:

- Into `sabench/transforms/samplewise.py`:
  - `t_temporal_log_cumsum`
  - `t_temporal_exceedance_duration`
  - `t_temporal_envelope`
  - `t_temporal_bandpass`
- Into `sabench/transforms/aggregation.py`:
  - `t_temporal_block_avg`

#### Slice: physical engineering transform extraction

Moved out of `sabench/transforms/transforms.py` into `sabench/transforms/engineering.py`:

- `t_carnot_quadratic`
- `t_arrhenius`
- `t_normalised_stress`

#### Slice: final transform implementation extraction

Moved the final transform implementation bodies out of `sabench/transforms/transforms.py`:

- Into `sabench/transforms/mathematical.py`:
  - `t_triangle_wave`
  - `t_signed_power`
  - `t_bernstein_b3`
  - `t_bimodal_flip`
  - `t_donut`
- Into `sabench/transforms/statistical.py`:
  - `t_yeo_johnson`

After this produced slice, there were zero transform registry entries pointing to `sabench.transforms.transforms`.

#### Slice: registry catalog ownership hardening

Added:

- `sabench/transforms/catalog.py`

Moved canonical catalog ownership out of `sabench/transforms/transforms.py`:

- `TRANSFORMS`
- `LINEAR_TRANSFORMS`
- `POINTWISE_TRANSFORMS`
- `AFFINE_TRANSFORMS`
- `NONLOCAL_TRANSFORMS`
- `CONVEX_TRANSFORMS`
- `CONCAVE_TRANSFORMS`
- `MONOTONE_TRANSFORMS`
- `NONMONOTONE_TRANSFORMS`
- `SMOOTH_TRANSFORMS`
- `NONSMOOTH_TRANSFORMS`

Updated:

- `sabench/transforms/registry.py` to import canonical catalog data from `sabench.transforms.catalog`, not `sabench.transforms.transforms`.
- `sabench/transforms/__init__.py` to export `TRANSFORMS` from the registry path.
- `tests/transforms/test_transform_registry_contract.py` to enforce this architecture.

#### Slice: retire the legacy transform monolith module

Added:

- `sabench/transforms/evaluation.py`

Moved public helper APIs out of `sabench/transforms/transforms.py`:

- `apply_transform`
- `score_transform`
- `_vw_s1`, renamed internally to `_variance_weighted_s1`

Deleted from the produced working tree:

- `sabench/transforms/transforms.py`

Updated:

- `sabench/transforms/__init__.py`
- `sabench/transforms/catalog.py`
- `tests/transforms/test_transform_registry_contract.py`
- `tests/transforms/test_transform_module_split.py`
- `tests/transforms/test_transform_utilities_layout.py`

The intended public API remains:

```python
from sabench.transforms import apply_transform, score_transform
```

but these should now come from `sabench.transforms.evaluation`, not from the retired monolith.

### Source bundle / transient-artifact hygiene hardening

The most recent produced slice hardened source bundle and cleanup behavior.

Updated:

- `.gitignore`
- `test_repo.sh`
- `scripts/build_source_bundle.py`
- `tests/integration/test_source_bundle_hygiene.py`
- `tests/integration/test_gate_contract_alignment.py`

Hardening intent:

- Source bundles exclude:
  - `diff.txt`
  - AppleDouble files like `._README.md`
  - `.coverage*`
  - `coverage.xml`
  - `*.zip`
  - `*.tar.gz`
  - `__MACOSX/`
  - caches and build artifacts
- `.gitignore` explicitly ignores:
  - `__MACOSX/`
  - `._*`
  - `*.zip`
  - `*.tar.gz`
  - `coverage.xml`
  - `diff.txt`
- `test_repo.sh --clean` removes:
  - `__MACOSX`
  - `._*`
  - `diff.txt`
  - `*.zip`
  - `*.tar.gz`

## Validation status from latest chat

Because the execution container lacked Pixi and its Python environment became unstable/hung on the scientific stack, the latest slices were not fully validated with Pixi or the full local gate in the container.

Validation that was claimed across slices included some combination of:

- `python -m compileall -q sabench tests scripts`
- `bash -n test_repo.sh`
- targeted pytest for transform split/registry/metadata tests where the environment allowed
- static architecture checks for imports, metadata module names, and absence of monolith references
- bundle hygiene checks by inspection/zip listing

Not completed in the container for later slices:

- full `pytest -q`
- `ruff`
- `mypy`
- Pixi-backed `test_repo.sh`
- full local gate

Therefore, the next chat should start by applying/inspecting the user's real repo and running the authoritative local Pixi gate before claiming completion.

## Produced changed-file bundles in the latest chat

These bundle names were produced sequentially:

1. `sabench_transform_spatial_dedup_slice.zip`
2. `sabench_transform_temporal_dedup_slice.zip`
3. `sabench_transform_physical_engineering_dedup_slice.zip`
4. `sabench_transform_final_monolith_dedup_slice.zip`
5. `sabench_registry_catalog_contract_slice.zip`
6. `sabench_retire_transform_monolith_slice.zip`
7. `sabench_source_bundle_hygiene_hardening_slice.zip`

Do not assume they were applied locally unless the user confirms. If a new repo archive is uploaded, ignore these as authoritative state and audit the new archive instead.

## Latest prioritized engineering manifest

### 1. Stale-state cleanup / repo stabilization

Status: partially addressed by the produced source-bundle hygiene slice.

Remaining work:

- Re-audit the actual live repo to confirm `.gitignore`, `test_repo.sh --clean`, and `scripts/build_source_bundle.py` contain the hardening changes.
- Run the relevant integration tests locally.
- Confirm generated bundles are repo-relative and clean.

### 2. Transform monolith endgame

Status: intended to be complete in produced slices.

Remaining work:

- Re-audit the live repo to confirm `sabench/transforms/transforms.py` is deleted.
- Confirm no imports reference `sabench.transforms.transforms`.
- Confirm no registry metadata entry points to `sabench.transforms.transforms`.
- Confirm public APIs `apply_transform` and `score_transform` still work from `sabench.transforms`.

### 3. Registry / metadata contract hardening

Status: partially addressed by `sabench/transforms/catalog.py` and registry contract tests.

Remaining work:

- Confirm canonical typed registries are the source of truth.
- Confirm metadata snapshots are generated/validated from canonical code definitions.
- Consider whether remaining legacy metadata files should be retired, explicitly documented as legacy, or validated against canonical exports.

### 4. Test and gate hardening

Status: not complete.

Remaining work:

- Run full Pixi-backed local gate.
- Fix any real failures from the user's local environment.
- Revisit whether CI/local gate alignment tests should be expanded now that the source bundle hardening exists.
- Confirm `test_repo.sh` executable bit is preserved in real Git.

### 5. Final packaging / release readiness

Status: not complete.

Remaining work:

- Package metadata audit.
- Public API audit.
- README/docs examples audit after monolith removal.
- Source bundle/release artifact smoke test.
- Final clean Git state and release checklist.

## Recommended next-chat objective

The next chat should not start with another blind architecture slice. It should start with a fresh audit of the actual live repo after the user's local application/validation state is known.

Suggested next action order:

1. Inspect uploaded live repo.
2. Determine Git/worktree state.
3. Confirm which of the seven latest changed-file slices are present.
4. Run compile/syntax checks.
5. Run targeted tests around:
   - transform registry contract
   - transform module split
   - transform utilities layout
   - registry metadata exports
   - source bundle hygiene
   - gate contract alignment
6. Run full pytest.
7. Run full Pixi-backed local gate.
8. Fix only real failures found in the live repo.
9. Then proceed to final release-readiness audit.

## Strong reminders for future work

- Do not assume generated bundles were applied.
- Do not assume this memory file is current.
- Do not patch from partial files or old bundles.
- Do not resurrect `sabench/transforms/transforms.py` unless the live repo proves the deletion was not applied and a narrower transition is required.
- Do not add compatibility shims.
- Do not package old zips, `.git`, coverage artifacts, caches, `__MACOSX`, AppleDouble files, or `diff.txt`.
- Use repo-relative bundle roots so `rsync` into `~/src/sabench` works cleanly.
- Include exact unzip, rsync, git add/rm, test, clean, commit, and push commands with every code/file handoff.
