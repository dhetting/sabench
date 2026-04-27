# MEMORY — sabench refactor status and hardening handoff (updated)

## User goals
- Finish the `sabench` refactor into a clean architecture centered on the two primary scientific concepts:
  - `benchmarks`
  - `transforms`
- Keep strict engineering discipline:
  - tests first
  - atomic slices only
  - no backward-compatibility shims
  - no hacks or workarounds
  - fix root causes
- Finalize and harden the repo for publication-quality packaging and gating.

## Non-negotiable workflow requirements
- Always inspect the **actual repo copy provided in the chat** before making further changes.
- Never assume a previously uploaded zip exactly matches the user's latest local checkout.
- Do not replace a live Git checkout with a source zip.
- In future deliverables, do **not** package `.git/`, `__pycache__/`, `.pytest_cache/`, coverage artifacts, or other transient files.
- Ensure `test_repo.sh` is executable in all future bundles.

## Important operational lesson from this conversation
Previous full-repo zip artifacts caused confusion because they looked like full repositories and, in at least one artifact, included `.git/` and transient cache/build files. This created Git-remote, auth, and rebase/conflict pain when the user applied them over a live local clone.

Going forward:
- treat zip handoffs as **source bundles only** unless explicitly told otherwise
- prefer changed-file bundles with repo-relative structure
- if giving a full repo zip, exclude:
  - `.git/`
  - `__pycache__/`
  - `.pytest_cache/`
  - `coverage.xml`
  - `.coverage*`
  - `dist/`
  - build artifacts
  - downloaded artifact zips
  - notebook runtime artifacts unless intentionally tracked

## Current architecture status (latest known from Slice 21)
### Benchmarks
The benchmark side is now substantially aligned with the target architecture:
- `sabench/benchmarks/` exists and is canonical
- families moved under:
  - `sabench/benchmarks/scalar/`
  - `sabench/benchmarks/spatial/`
  - `sabench/benchmarks/functional/`
- `sabench/_base.py` is gone
- `sabench/benchmarks/base.py` is canonical
- typed benchmark registry exists in `sabench/benchmarks/registry.py`
- typed benchmark metadata/spec exists in `sabench/benchmarks/types.py`
- typed benchmark registry was expanded to the full 29-benchmark catalogue
- canonical benchmark registry metadata export exists:
  - `sabench/metadata/benchmarks_registry_metadata.json`

### Transforms
The transform side is partially modularized and materially improved, but not fully finalized:
- typed transform registry exists in `sabench/transforms/registry.py`
- typed transform metadata/spec exists in `sabench/transforms/types.py`
- typed transform registry was expanded to the full 172-transform catalogue
- canonical transform registry metadata export exists:
  - `sabench/metadata/transforms_registry_metadata.json`
- transform property sets exported from `sabench.transforms` are now canonical and registry-derived, not re-exported from hand-maintained legacy sets
- focused transform modules now exist:
  - `pointwise.py`
  - `linear.py`
  - `nonlinear.py`
  - `samplewise.py`
  - `aggregation.py`
  - `field_ops.py`
  - `utilities.py`
  - `base.py`
  - `types.py`
  - `registry.py`
- extracted transform implementations now include at least:
  - `t_affine` → `linear.py`
  - `t_tanh_pointwise` → `pointwise.py`
  - `t_square_pointwise` → `pointwise.py`
  - `t_exp_pointwise` → `pointwise.py`
  - `t_relu_pointwise` → `pointwise.py`
  - `t_log1p_abs` → `pointwise.py`
  - `t_sqrt_abs` → `pointwise.py`
  - `t_abs_pointwise` → `pointwise.py`
  - `t_softplus_pointwise` → `nonlinear.py`
  - `t_temporal_cumsum` → `samplewise.py`
  - `t_temporal_peak` → `aggregation.py`
  - `t_gradient_magnitude` → `field_ops.py`
- `sabench/transforms/transforms.py` still exists and still contains most of the transform catalogue; the extracted subset is delegated/imported from focused modules

### Root API
- `sabench/__init__.py` has been narrowed to a minimal root API
- README and notebook examples were updated to use scientific subpackages instead of wide root re-exports

### Tests
- tests were moved out of the runtime package
- top-level test layout now includes:
  - `tests/benchmarks/`
  - `tests/transforms/`
  - `tests/analysis/`
  - `tests/sampling/`
  - `tests/integration/`
- `sabench/tests/` was removed
- tooling was updated accordingly (`pyproject.toml`, `pixi.toml`, `.pre-commit-config.yaml`, `test_repo.sh`)

### Metadata
- legacy metadata files still exist and were preserved:
  - `sabench/metadata/benchmarks_metadata.json`
  - `sabench/metadata/transforms_metadata.json`
- canonical metadata export code exists in:
  - `sabench/metadata/exports.py`
- drift tests were added so canonical registry-generated JSON must match committed snapshots

## Validation status last claimed in conversation
Repeatedly validated in the container with:
- `python -m compileall sabench tests`
- targeted pytest
- full pytest
- coverage
- `bash -n test_repo.sh`

Repeated blocker in the container:
- `pixi` was not installed, so `./test_repo.sh --check-only --no-notebook` could not be fully executed there

Therefore, a key next-chat goal is to validate the **actual repo** in an environment where Pixi is available.

## Known fixes already made during the refactor
- benchmark spec flags renamed to snake_case (`has_analytical_s1`, `has_analytical_st`)
- hardcoded repo-root assumptions removed from new layout tests; tests should derive paths from the installed package root
- `test_repo.sh` trailing-whitespace count bug fixed (`0\n0` arithmetic issue under `pipefail`)
- transform registry typing fixes applied for mypy:
  - typed legacy metadata view
  - typed binding wrapper instead of problematic `functools.partial`
  - aliasing of legacy imported property sets to avoid mypy `no-redef`
- `test_repo.sh` executability was explicitly requested by the user and should always be preserved

## Known problems / caution flags still relevant
1. **Full repo zip hygiene is not yet trustworthy**
   - At least one artifact included `.git/`, `__pycache__/`, `.pytest_cache/`, and other transient files.
   - Hardening should add a packaging/archiving rule and tests/scripts that prevent this.

2. **The transform monolith still exists**
   - `sabench/transforms/transforms.py` is still large and still canonical for many actual function definitions.
   - The next serious architecture work is to keep extracting transforms by mechanism until the monolith can be removed.

3. **Legacy metadata files still exist beside canonical registry exports**
   - Canonical exports now exist, but the repo still contains older hand-maintained JSON files.
   - A future decision is needed:
     - either keep legacy files temporarily with explicit status
     - or migrate docs/tests/users fully to canonical exports and retire the legacy duplicates

4. **Actual local gate status remains to be proven in the user’s environment**
   - Need a full run of local gate / Pixi / mypy / pre-commit in the live checkout
   - Need to fix only real remaining failures from that environment

5. **Git/rebase conflict pain occurred late in the conversation**
   - A later chat should prioritize stabilizing the user's actual checkout and avoiding further workflow disruption.
   - Do not create another big replacement zip as the default mechanism.

## Best understanding of latest user pain point
The user ended up in a messy rebase/conflict state while trying to apply later slices locally. They asked for conflict-fixed files directly. This means the next chat should start by stabilizing the **real local working tree** and then move to final hardening, not by continuing blindly with more refactor slices.

## Recommended next-chat priorities
Order matters.

### Priority 1 — stabilize the real local checkout
- inspect the actual uploaded repo copy or user-provided conflict state
- confirm whether rebase/merge is still in progress
- resolve only the real remaining conflicts/issues in the live tree
- do not replace the repo with a full zip

### Priority 2 — harden packaging and artifact hygiene
Add or fix a deterministic packaging/export path so future bundles:
- exclude `.git/`
- exclude caches and coverage artifacts
- preserve executable bits such as `test_repo.sh`
- are safe to apply into an existing clone

### Priority 3 — run the true local gate
In the user’s environment, run and fix as needed:
- `./test_repo.sh --check-only --no-notebook`
- or the repo’s current authoritative gate command
- full `mypy`, `ruff`, `pytest`, coverage, and any Pixi tasks

### Priority 4 — continue transform monolith decomposition only after the repo is stable
Suggested next architecture work after hardening:
- continue extracting coherent transform groups by mechanism
- add tests first for each subset
- remove duplicate definitions from the monolith as each subset moves
- eventually delete `sabench/transforms/transforms.py`

## Strong prompt-level reminders for the next chat
- Inspect the actual provided repo first.
- Do not assume the latest zip is the user’s truth.
- Do not hand off a repo zip that includes `.git/` or caches.
- Do not tell the user to replace their clone with a full zip.
- Prefer changed-file bundles and exact overwrite commands.
- Keep slices atomic and green.
- Fix only the real issue in front of you before advancing architecture further.

