# MEMORY — sabench release-readiness continuation

Last updated: 2026-04-29

This file is continuity context only. It is advisory, not authoritative. In any new
chat or future slice, audit the live repo on disk before trusting this file,
previous bundles, prior assistant claims, or claimed local state.

## Current working posture

The `sabench` repo is in late-stage release-readiness hardening. The main
transform refactor is believed to be complete: the legacy transform monolith has
been retired, transform implementations live in focused modules, canonical
transform catalog data lives in `sabench/transforms/catalog.py`, and transform
evaluation helpers live in `sabench/transforms/evaluation.py`.

The remaining work is release hygiene, packaging verification, documentation/API
consistency, final gate confirmation, and final handoff. Do not resume transform
extraction unless a fresh live-repo audit finds duplicated implementations or
stale monolith references.

## Non-negotiable operating rules

- Treat the live repo as the only source of truth.
- Audit first, then patch.
- Strict test-driven development.
- Atomic slices only.
- No backward-compatibility shims unless explicitly requested.
- No hacks or workarounds.
- Fix root causes.
- Tests belong outside the runtime package.
- Metadata must be generated from, or validated against, canonical code
  definitions.
- Do not claim validation that was not actually run.
- Use Pixi-backed repo gates when available.
- Future bundles must contain repo-relative paths directly at archive root so
  `rsync` into `~/src/sabench` works cleanly.

## Handoff format the user expects

For each completed slice, provide:

1. changed files list,
2. validation actually run,
3. bundle download link if artifact download works,
4. unzip + `rsync` commands from `~/Downloads` into `~/src/sabench`,
5. separate `git add` / `git rm` commands,
6. repo test commands,
7. clean command,
8. commit and push commands.

If artifact generation/download is unreliable, stop regenerating repeatedly and
provide raw full-file contents or copy-pasteable patch scripts.

## Verified transform architecture from the uploaded `sabench-v1.zip` audit

The uploaded repo snapshot used for the recent hardening work showed:

- `sabench/transforms/transforms.py` was absent.
- No runtime imports should reference `sabench.transforms.transforms`.
- Tests now explicitly guard against stale monolith references.
- `sabench.transforms.__init__` exports `apply_transform` and
  `score_transform`.
- `apply_transform` and `score_transform` are implemented in
  `sabench.transforms.evaluation`.
- `sabench/transforms/catalog.py` owns canonical raw transform catalog data.
- Focused transform modules include aggregation, ecological, engineering,
  environmental, field operations, financial, mathematical, pointwise,
  samplewise, and statistical groups.
- Registry metadata is expected to resolve to live registered objects and must
  not point back to the removed monolith.

Re-verify these points in any new chat before making further claims.

## Recent hardening slices completed in this chat sequence

The following slices were produced after the `sabench-v1.zip` audit. Some were
validated by the user locally/CI during the sequence; nevertheless, a new chat
must inspect the live repo and run the gate again.

### 1. Recursive clean hardening

Files:

- `test_repo.sh`
- `tests/integration/test_gate_contract_alignment.py`

Intent:

- Make `test_repo.sh --clean` remove nested bundle/platform artifacts
  recursively rather than only root-level artifacts.
- Remove nested `__MACOSX`, `.DS_Store`, AppleDouble `._*`, `diff.txt`, `*.zip`,
  and `*.tar.gz` while preserving `.git` and `.pixi`.
- Add integration tests that assert the recursive cleanup contract.

### 2. Build metadata hygiene

Files:

- `scripts/build_source_bundle.py`
- `test_repo.sh`
- `tests/integration/test_source_bundle_hygiene.py`
- `tests/integration/test_gate_contract_alignment.py`

Intent:

- Exclude root and nested `*.egg-info/` directories from source bundles.
- Recursively clean `dist/`, `build/`, and `*.egg-info/` during clean/build
  stages.
- Add source-bundle and gate-contract regression tests for generated package
  metadata.

### 3. Tracked artifact hygiene

Files:

- `tests/integration/test_gate_contract_alignment.py`

Deleted generated/local artifacts:

- `.coverage`
- `coverage.xml`
- `REMOVE_FILES.txt`
- `REMOVED_PATHS.txt`
- `sabench-v0.3.zip`

Intent:

- Prevent generated artifacts, local handoff files, build products, old archives,
  caches, and platform detritus from being tracked by git.

### 4. Actual source-bundle regression coverage

Files:

- `tests/integration/test_source_bundle_hygiene.py`

Intent:

- Build and inspect an actual source bundle from the real repo root, not only
  synthetic fixtures.
- Assert the actual bundle uses repo-relative paths and excludes generated,
  cached, platform, archive, build, and package metadata artifacts.

### 5. Registry/metadata consistency hardening

Files:

- `tests/integration/test_registry_metadata_exports.py`

Intent:

- Verify benchmark metadata resolves to importable registered classes.
- Verify transform metadata resolves to importable canonical catalog functions.
- Verify committed metadata snapshots contain no stale
  `sabench.transforms.transforms` references.

Follow-up fix:

- Benchmark metadata `name` is a display name, not necessarily the registry key.
  The test should require a non-empty display name and validate class/module
  identity against the registry, but should not assert display name equals the
  registry key.

### 6. Built-wheel package smoke

Files:

- `.github/workflows/ci.yml`
- `pixi.toml`
- `test_repo.sh`
- `scripts/check_built_distribution.py`
- `tests/integration/test_gate_contract_alignment.py`

Intent:

- Add a `package-smoke` Pixi task.
- Run `package-smoke` in local gate and GitHub Actions build job after build and
  `twine check`.
- Verify exactly one wheel and one sdist exist in `dist/`.
- Install the built wheel and import `sabench` from outside the source package
  tree.
- Verify distribution metadata version matches `sabench.__version__`.
- Verify the installed runtime package excludes tests and the removed transform
  monolith.
- Smoke-test installed benchmark/transform registries and public transform API.

Follow-up fix:

- The first path check incorrectly rejected imports from `.pixi/envs/ci` because
  that directory is under the repo root. The correct check rejects imports from
  the source package tree `repo_root / "sabench"`, while allowing valid installed
  package imports from `.pixi/envs/ci/lib/.../site-packages`.

### 7. README/notebook/API consistency hardening

Files:

- `tests/integration/test_root_api_narrow.py`

Intent:

- Keep the root `sabench` API intentionally narrow.
- Require README and demo notebook examples to use subpackage imports rather than
  root imports.
- Verify README benchmark/transform/property counts match live registries and
  metadata.
- Verify demo notebook transform keys and catalog/property counts match the live
  registry.

Follow-up fix:

- Remove an invalid f-string with no interpolation from
  `tests/integration/test_root_api_narrow.py`.

### 8. Memory update slice

Files:

- `docs/MEMORY.md`

Intent:

- Replace stale pre-hardening continuity notes with this current release-readiness
  handoff.
- Preserve the warning that memory is advisory only.

## Current likely git state after applying recent slices

Expected changed/removed paths before committing the complete hardening sequence
may include:

- `.github/workflows/ci.yml`
- `pixi.toml`
- `test_repo.sh`
- `docs/MEMORY.md`
- `scripts/build_source_bundle.py`
- `scripts/check_built_distribution.py`
- `tests/integration/test_gate_contract_alignment.py`
- `tests/integration/test_registry_metadata_exports.py`
- `tests/integration/test_root_api_narrow.py`
- `tests/integration/test_source_bundle_hygiene.py`
- deleted `.coverage`
- deleted `coverage.xml`
- deleted `REMOVE_FILES.txt`
- deleted `REMOVED_PATHS.txt`
- deleted `sabench-v0.3.zip`

Do not blindly stage this list. Run `git status --short` in the live repo and
stage only what is actually present and intended.

## Validation commands to prefer

Use the repo-defined gate if it differs, but start with:

```bash
cd ~/src/sabench
python -m compileall -q sabench tests scripts
bash -n test_repo.sh
PYTHONPATH=. pytest -q tests/integration/test_gate_contract_alignment.py --tb=short
PYTHONPATH=. pytest -q tests/integration/test_source_bundle_hygiene.py --tb=short
PYTHONPATH=. pytest -q tests/integration/test_registry_metadata_exports.py --tb=short
PYTHONPATH=. pytest -q tests/integration/test_root_api_narrow.py --tb=short
PYTHONPATH=. pytest -q
./test_repo.sh --check-only --no-notebook
```

Direct package-smoke path:

```bash
cd ~/src/sabench
rm -rf dist build *.egg-info
pixi run -e ci build
pixi run -e ci twine-check
pixi run -e ci package-smoke
```

Full gate when ready:

```bash
cd ~/src/sabench
./test_repo.sh --check-only
```

Clean command:

```bash
cd ~/src/sabench
./test_repo.sh --clean --no-notebook
```

## Remaining high-priority tasks

1. Apply this memory update and run the targeted docs/memory or repo-style checks
   available in the live repo.
2. Re-run the package-smoke path after the Pixi source-tree path fix.
3. Run the full local gate, including notebooks if feasible.
4. Confirm GitHub Actions passes after the package-smoke and docs/API slices.
5. Build and inspect the final source bundle from a clean working tree.
6. Confirm `git status --short` contains only intentional changes, then commit.
7. Create the final release-readiness handoff or tag/release checklist.

## Suggested final commit messages

Use separate commits if the slices are not yet committed:

```bash
git commit -m "chore(gate): clean bundle artifacts recursively"
git commit -m "chore(gate): harden build metadata hygiene"
git commit -m "chore(repo): remove tracked generated artifacts"
git commit -m "test(bundle): validate actual source bundle hygiene"
git commit -m "test(metadata): verify registry export references"
git commit -m "fix(ci): allow wheel smoke from pixi environment"
git commit -m "test(docs): verify public API documentation stays current"
git commit -m "docs: update release-readiness memory"
```

If all slices are intentionally squashed together, use a single commit message
such as:

```bash
git commit -m "chore: finalize release-readiness hardening"
```
