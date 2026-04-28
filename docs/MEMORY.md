# MEMORY — sabench refactor/hardening continuation

## Current status summary

The `sabench` repo is in late-stage refactor hardening/finalization.

The core transform refactor appears mostly complete based on the latest uploaded repo audit, but the repo is **not yet release-ready**. Remaining work is primarily reproducibility, gate hygiene, source-bundle hygiene, metadata/registry validation, CI/local contract alignment, and final release-readiness review.

Do **not** assume any generated bundle, previous patch, or assistant claim was actually applied locally. The user has had repeated artifact download failures. Treat the fresh repo copy uploaded in the next chat as the only authoritative source of truth.

## Core operating rule

Do **not** trust prior bundles, summaries, generated files, claims about local state, or this memory file as authoritative.

Always begin from the actual live repo copy uploaded in the current chat.

Audit first, then patch.

## User workflow preferences

- Strict test-driven development.
- Atomic slices only.
- No backward-compatibility shims.
- No hacks/workarounds.
- Fix root causes.
- Tests must live outside the runtime package.
- Prefer exhaustive tests over minimal tests.
- Run syntax/compile checks, targeted tests, full pytest, and repo gate as far as the environment allows.
- Do not claim full validation unless it actually ran.
- If artifact downloads fail, provide raw full-file contents or raw patch scripts instead of repeatedly generating broken bundles.
- For normal handoffs, provide:
  - downloadable bundle,
  - unzip + `rsync` commands from `~/Downloads` into `~/src/sabench`,
  - separate `git add` / `git rm` commands,
  - test commands,
  - clean commands,
  - commit and push commands.
- Future bundles must have repo-relative paths directly at the archive root.

## Latest known repo/refactor state from uploaded `sabench-v1.zip`

A fresh audit of the uploaded repo indicated:

- `sabench/transforms/transforms.py` was absent.
- Focused transform modules existed, including:
  - `aggregation.py`
  - `catalog.py`
  - `engineering.py`
  - `evaluation.py`
  - `field_ops.py`
  - `financial.py`
  - `mathematical.py`
  - `statistical.py`
- `sabench.transforms.__init__` appeared to export `apply_transform` and `score_transform` from `sabench.transforms.evaluation`.
- `sabench/transforms/catalog.py` appeared to own canonical raw transform catalog data via `TRANSFORMS`.
- No inspected metadata/tests referenced `sabench.transforms.transforms`.
- Prior handoff indicated the financial/risk transform family had been extracted into `sabench.transforms.financial`, but this must still be verified in the fresh repo.

Important: these findings describe one uploaded repo snapshot only. Verify again from the fresh upload.

## Known hygiene problem in latest uploaded archive

The latest uploaded `sabench-v1.zip` itself was contaminated and should not be treated as a clean source bundle. It contained transient/generated/local artifacts, including:

- `.git/`
- `.coverage`
- `coverage.xml`
- `__MACOSX/`
- `__pycache__/`
- `.pyc` files
- duplicate archive entries
- evidence of non-clean git state such as `.git/REBASE_HEAD`

This strongly suggests source-bundle and local-clean hygiene remain high-priority.

## Attempted but not confirmed applied: source-bundle path-safety slice

A proposed hardening slice targeted:

- `scripts/build_source_bundle.py`
- `tests/integration/test_source_bundle_hygiene.py`

Intent of the slice:

- Skip symlinks during recursive source bundle construction.
- Reject absolute `--path` arguments.
- Reject selected paths that resolve outside `--source-root`.
- Reject explicitly selected symbolic links.
- Preserve exclusions for transient artifacts:
  - `.git`
  - `.pixi`
  - `__MACOSX`
  - `__pycache__`
  - `.pytest_cache`
  - `.mypy_cache`
  - `.ruff_cache`
  - `dist`
  - `build`
  - `.DS_Store`
  - `.coverage*`
  - `coverage.xml`
  - `diff.txt`
  - AppleDouble `._*`
  - `.pyc`
  - `.pyo`
  - `.whl`
  - `.zip`
  - `.tar.gz`

Tests proposed:

- `test_source_bundle_excludes_transient_repo_artifacts_and_preserves_executable_bits`
- `test_source_bundle_can_build_changed_file_bundle`
- `test_source_bundle_skips_symbolic_links`
- `test_changed_file_bundle_rejects_selected_paths_outside_source_root`

Important: The downloadable bundle failed repeatedly. The assistant then gave raw full-file contents for:

- `scripts/build_source_bundle.py`
- `tests/integration/test_source_bundle_hygiene.py`

Do **not** assume those file contents were applied. In the next chat, inspect the live files first and compare.

## Attempted but not confirmed applied: recursive clean gate slice

A second proposed hardening slice targeted:

- `test_repo.sh`
- `tests/integration/test_gate_contract_alignment.py`

Intent of the slice:

- Replace root-only cleanup in `test_repo.sh`:

```bash
rm -rf __MACOSX
rm -f ._.* diff.txt *.zip *.tar.gz
```

with recursive cleanup that removes nested transient bundle/platform artifacts while excluding `.git` and `.pixi`.

Proposed replacement:

```bash
info "Removing transient bundle/platform artefacts"
find . -type d -name "__MACOSX" \
  -not -path "./.git/*" -not -path "./.pixi/*" \
  -prune -exec rm -rf {} +
find . -type f \( -name ".DS_Store" -o -name "._*" -o -name "diff.txt" -o -name "*.zip" -o -name "*.tar.gz" \) \
  -not -path "./.git/*" -not -path "./.pixi/*" \
  -delete
```

Proposed test change:

- Replace `test_clean_stage_removes_local_bundle_and_platform_artifacts`
- With `test_clean_stage_removes_local_bundle_and_platform_artifacts_recursively`
- Require recursive `find` commands in the clean section.
- Assert old root-only `rm` commands are absent.

Validation claimed by assistant in broken environment:

```bash
bash -n test_repo.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest -q tests/integration/test_gate_contract_alignment.py --tb=short
timeout 30 python -m compileall -q sabench tests scripts
```

Important: The bundle handoff for this slice also failed. Do **not** assume this slice was applied locally.

## Suggested immediate next priority

Start with a fresh audit of the actual repo uploaded in the new chat, then determine whether either attempted hygiene slice has actually been applied.

Recommended next sequence:

1. Inspect repo tree and actual files.
2. Determine real git/worktree state.
3. Read `docs/MEMORY.md` as advisory only.
4. Verify transform refactor state:
   - whether `sabench/transforms/transforms.py` exists,
   - whether any imports reference `sabench.transforms.transforms`,
   - whether metadata points to deleted monolith,
   - whether `catalog.py` owns raw catalog data,
   - whether `evaluation.py` owns `apply_transform` / `score_transform`,
   - whether `sabench.transforms.__init__` exports the intended public API.
5. Verify source-bundle hygiene state:
   - inspect `scripts/build_source_bundle.py`,
   - inspect `tests/integration/test_source_bundle_hygiene.py`,
   - check symlink/path-escape behavior if not tested.
6. Verify recursive clean gate state:
   - inspect `test_repo.sh`,
   - inspect `tests/integration/test_gate_contract_alignment.py`,
   - check whether nested artifact cleanup is tested.
7. Run validation as far as available:
   - `python -m compileall -q sabench tests scripts`
   - `bash -n test_repo.sh`
   - relevant targeted tests
   - full `PYTHONPATH=. pytest -q`
   - `./test_repo.sh --check-only --no-notebook`
8. Choose exactly one highest-value atomic slice based on live failures.
9. Write failing tests first.
10. Implement only that slice.
11. Provide handoff using raw full-file contents if artifact download is unreliable.

## Current likely remaining work

The transform architecture likely no longer needs major extraction work unless the fresh audit finds drift. Remaining work is probably:

1. Clean-state/source-bundle hygiene.
2. Recursive local-clean behavior.
3. Gate/CI/local contract alignment.
4. Registry/metadata contract hardening.
5. README/docs/API release-readiness review.
6. Final source bundle generation and release checklist.

## Validation commands to prefer

Use repo-defined commands if they differ, but start with:

```bash
cd ~/src/sabench
python -m compileall -q sabench tests scripts
bash -n test_repo.sh
PYTHONPATH=. pytest -q tests/integration/test_source_bundle_hygiene.py --tb=short
PYTHONPATH=. pytest -q tests/integration/test_gate_contract_alignment.py --tb=short
PYTHONPATH=. pytest -q
./test_repo.sh --check-only --no-notebook
```

For cleanup:

```bash
cd ~/src/sabench
./test_repo.sh --clean --no-notebook
```

## Handoff format

When a slice is complete, provide:

1. Changed files list.
2. Validation actually run.
3. Downloadable bundle if artifact system works.
4. If artifact system is unreliable, provide raw full-file contents instead.
5. Commands:

```bash
cd ~/Downloads
rm -rf <bundle_name>
mkdir -p <bundle_name>
unzip -o <bundle_name>.zip -d <bundle_name>
rsync -av ~/Downloads/<bundle_name>/ ~/src/sabench/
```

Then:

```bash
cd ~/src/sabench
python -m compileall -q sabench tests scripts
bash -n test_repo.sh
PYTHONPATH=. pytest -q <targeted tests>
PYTHONPATH=. pytest -q
./test_repo.sh --check-only --no-notebook
```

Then:

```bash
cd ~/src/sabench
git add <changed files>
git rm <removed files, if any>
git commit -m "<message>"
git push
```
