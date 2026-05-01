# Copilot instructions for sabench

## Purpose

This repository is a scientific Python package for benchmark and transformation experiments in sensitivity analysis, Sobol-index behavior, and transformation noncommutativity.

Operate as a careful repository-local engineering assistant. Prioritize correctness, reproducibility, scientific traceability, and clean integration with the existing test and release gates.

## Source-of-truth policy

Treat the live repository checkout as the only source of truth.

Do not trust prior summaries, generated bundles, assistant claims, stale memory files, stale manifests, or assumed local state. Repository memory files and planning documents may provide useful context, but they are advisory, not authoritative.

Before changing code, inspect the exact live files and tests that define the current behavior.

## Repository structure

The canonical package structure is:

- `sabench.benchmarks`: benchmark implementations and typed benchmark registries.
- `sabench.transforms`: transformation implementations and typed transform registries.
- `sabench.metadata`: generated or validated registry metadata snapshots.
- `sabench.analysis`: reusable scientific analysis utilities, estimators, metrics, and grid engines.
- `notebooks`: deterministic analysis notebooks that orchestrate tested package utilities.
- `tests`: test suite; tests must remain outside the runtime package.
- `scripts`: repository, source-bundle, package, and release automation.
- `docs`: documentation, memory files, engineering manifests, and manuscript-facing context.
- `docs/manuscripts`: local manuscript and memo sources used for scientific context, formulas, terminology, and validation targets.

Do not reintroduce removed legacy modules, monoliths, compatibility aliases, or wrapper paths unless explicitly requested by the maintainer.

## Manuscript and memo context

The folder `docs/manuscripts` may contain local LaTeX manuscripts, memos, or related scientific documents. Treat these files as contextual sources for scientific definitions, formulas, notation, and terminology.

When implementing methods that are described in manuscripts or memos:

- inspect the relevant source documents directly;
- preserve the manuscript’s mathematical meaning;
- keep notation and terminology consistent with the manuscript where practical;
- implement formulas in tested Python modules, not only in notebooks;
- add tests that verify the formula behavior on simple, interpretable examples;
- clearly separate theorem-supported calculations from empirical diagnostics;
- avoid claiming a theorem guarantee when the code only computes a sample-based diagnostic.

Do not modify manuscript files unless the task explicitly asks for manuscript editing.

## Scientific implementation principles

Keep scientific logic in reusable, tested package modules. Notebooks should orchestrate and present tested functionality rather than contain core scientific implementations.

Prefer registry-driven workflows. Benchmark and transform inventories should come from live registries rather than hard-coded lists.

Preserve typed registry contracts. Registry definitions, metadata snapshots, public exports, and validation tests must remain aligned.

When adding analysis functionality:

- define small typed data structures where useful;
- validate inputs explicitly;
- return structured results that can be converted into tidy tables;
- handle pair-level failures as explicit statuses rather than crashing entire grid analyses;
- distinguish unsupported, invalid, failed, diagnostic-only, and computed cases;
- include enough metadata in result rows for reproducibility and audit.

## Notebook policy

Notebooks are production artifacts.

They must be:

- deterministic by default;
- executable from a clean checkout;
- clean in git, with no committed outputs and no execution counts;
- driven by public or intentionally exported package APIs;
- configured through visible configuration cells and, where appropriate, environment variables;
- capable of running in a small CI-safe mode;
- capable of exporting deterministic tables or figures to ignored output directories.

Do not hide scientific logic in notebooks. Move reusable computations into `sabench.analysis` or another appropriate tested module.

Notebook outputs must not be committed unless the maintainer explicitly asks for committed outputs.

## Testing discipline

Use strict test-driven development.

Before implementing behavior, write or update tests that express the intended contract.

Do not weaken tests to make broken code pass. If a test is wrong, fix the test by aligning it with the live registry, documented contract, or verified scientific requirement.

Add tests at the right level:

- unit tests for formulas, validators, and small utilities;
- analysis tests for metric engines, estimators, and grid engines;
- integration tests for registries, metadata, notebooks, package build behavior, source-bundle hygiene, and gate alignment;
- notebook-contract tests for notebook cleanliness, execution, exports, and schema checks.

Tests should be deterministic. Use fixed seeds for stochastic workflows.

Avoid tests that depend on local user-specific paths, uncommitted generated outputs, network access, or hidden state.

## Validation discipline

Prefer the repository-defined validation gate.

For normal code/test changes, run at minimum:

    python -m compileall -q sabench tests scripts
    bash -n test_repo.sh
    PYTHONPATH=. pytest -q tests/analysis --tb=short
    PYTHONPATH=. pytest -q tests/integration --tb=short
    ./test_repo.sh --check-only --no-notebook

For notebook changes, also run relevant notebook-contract tests and the full notebook gate when feasible:

    PYTHONPATH=. pytest -q tests/integration --tb=short
    ./test_repo.sh --check-only

For package/release changes, also run the package build and smoke checks defined by the repository, typically through Pixi tasks.

If a tool or gate cannot be run, report exactly what was not run and why. Do not claim full validation unless the full relevant gate actually passed.

## Pixi and environment policy

Pixi is the authoritative environment manager when the repo defines Pixi tasks.

Prefer repo-defined Pixi tasks and shell gates over ad hoc commands. If local and CI behavior differ, treat that as a bug to investigate.

Do not add dependencies casually. If a new dependency is necessary, update the appropriate Pixi and project metadata files, then verify local and CI behavior.

## Git and branch policy

Use one branch and one pull request per atomic slice.

Branch names should be short and descriptive, for example:

- `feat/<short-feature-name>`
- `fix/<short-fix-name>`
- `test/<short-test-name>`
- `docs/<short-doc-change>`
- `refactor/<short-refactor-name>`

Start branches from updated `main`.

Do not mix unrelated work in a single branch. Avoid broad drive-by edits.

Before committing:

- inspect `git status --short`;
- inspect `git diff`;
- run relevant validation;
- ensure generated artifacts are not staged;
- stage only intended files.

Commit messages should be concise and conventional, for example:

- `feat(analysis): add bounds grid engine`
- `test(notebooks): validate noncommutativity exports`
- `fix(metadata): align registry export paths`
- `docs: update analysis notebook documentation`

## Pull request policy

Each PR should describe:

- the purpose of the slice;
- changed files;
- tests and validation actually run;
- any validation not run and why;
- known limitations or follow-up tasks.

Do not merge until local validation and GitHub CI pass.

If a PR is superseded, close it and delete the branch rather than letting stale branches accumulate.

## Generated artifact policy

Do not commit or package generated, transient, platform, or build artifacts.

Keep these out of git and source bundles:

- `.git`
- `.pixi`
- `__pycache__`
- `.pytest_cache`
- `.ruff_cache`
- `.mypy_cache`
- `.coverage`
- `coverage.xml`
- `htmlcov`
- `dist`
- `build`
- `*.egg-info`
- `*.zip`
- `*.tar.gz`
- `*.whl`
- `__MACOSX`
- `.DS_Store`
- `._*`
- `diff.txt`
- temporary handoff files
- notebook execution outputs unless explicitly requested

Source bundles should contain repo-relative paths only. They must not include symlinks, path escapes, caches, coverage files, build artifacts, old archives, AppleDouble files, platform metadata, or local environment directories.

## Style and formatting

Follow the repository’s configured style tools. Do not invent a separate formatting convention.

Prefer readable, typed Python with explicit validation and small functions.

Keep line lengths, import ordering, docstrings, and notebook formatting consistent with the existing repository configuration.

Do not suppress lint/type/test failures unless the suppression is justified, narrow, and tested.

## Public API policy

Keep the public package API intentionally narrow.

Do not add root-level exports casually. Prefer subpackage imports unless the project already exposes a stable public symbol intentionally.

When changing public exports:

- update tests that define the public API contract;
- update README or notebook examples if needed;
- update metadata validation if registry-visible behavior changes;
- verify package smoke tests still pass from an installed wheel.

## Registry and metadata policy

Benchmark and transform registries are canonical integration surfaces.

When editing benchmark or transform definitions:

- update or verify typed registry entries;
- update or verify generated metadata snapshots;
- verify metadata-export tests;
- ensure module paths are importable;
- ensure stale paths are not retained;
- ensure notebooks and examples remain registry-driven.

Do not manually patch metadata snapshots without understanding the generator/validation path.

## Error-handling policy

Prefer explicit validation and typed statuses over broad exception swallowing.

Grid analyses should capture expected pair-level failures as structured statuses where appropriate. Unexpected programming errors should still fail tests.

Do not hide broken scientific calculations behind generic fallback values.

## Documentation policy

Documentation should describe current, validated behavior.

Do not claim a feature is complete unless it is implemented and tested.

Keep README, notebooks, engineering manifests, and memory files consistent with the live package behavior.

When updating docs, avoid stale hard-coded counts unless tests verify that those counts match live registries.

## Final handoff expectations

When finishing a slice, report:

- branch name;
- PR number or URL if created;
- changed files;
- what changed;
- validation actually run;
- validation not run and why;
- any generated artifacts intentionally left uncommitted;
- remaining risks;
- next recommended atomic slice.

Be precise. Do not overstate validation or readiness.
