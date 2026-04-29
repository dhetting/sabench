import subprocess
from pathlib import Path, PurePosixPath

import pytest
import tomllib

import sabench


def _repo_root() -> Path:
    return Path(sabench.__file__).resolve().parent.parent


def _load_pixi_manifest() -> dict:
    with (_repo_root() / "pixi.toml").open("rb") as fh:
        return tomllib.load(fh)


def _clean_section() -> str:
    script = (_repo_root() / "test_repo.sh").read_text(encoding="utf-8")
    return script.split("# STAGE 1 — INSTALL / SYNC ENVIRONMENT")[0]


def test_pixi_ci_environment_defines_authoritative_gate_tasks() -> None:
    manifest = _load_pixi_manifest()
    tasks = manifest["tasks"]
    for task_name in [
        "lint",
        "fmt-check",
        "typecheck",
        "test-cov",
        "build",
        "twine-check",
        "package-smoke",
    ]:
        assert task_name in tasks


def test_test_repo_delegates_check_and_build_stages_to_pixi_tasks() -> None:
    script = (_repo_root() / "test_repo.sh").read_text(encoding="utf-8")
    assert 'PIXI="pixi run -e ci"' in script
    for task_name in [
        "lint",
        "fmt-check",
        "typecheck",
        "test-cov",
        "build",
        "twine-check",
        "package-smoke",
    ]:
        assert f"$PIXI {task_name}" in script

    check_and_build_section = script.split("# STAGE 2 — PREPARE")[1]
    assert 'record "ruff lint" $PIXI lint' in check_and_build_section
    assert 'record "ruff format --check" $PIXI fmt-check' in check_and_build_section
    assert 'record "mypy" $PIXI typecheck' in check_and_build_section
    assert 'record "pytest + coverage" $PIXI test-cov' in check_and_build_section
    assert 'record "python -m build" $PIXI build' in check_and_build_section
    assert 'record "twine check" $PIXI twine-check' in check_and_build_section
    assert 'record "package import smoke" $PIXI package-smoke' in check_and_build_section

    assert 'record "ruff lint" $PIXI ruff check sabench tests' not in check_and_build_section
    assert (
        'record "ruff format --check" $PIXI ruff format --check sabench tests'
        not in check_and_build_section
    )
    assert 'record "mypy" $PIXI mypy sabench' not in check_and_build_section
    assert 'record "python -m build" $PIXI python -m build' not in check_and_build_section
    assert 'record "twine check" $PIXI twine check --strict dist/*' not in check_and_build_section


def test_package_smoke_task_installs_built_wheel_outside_source_tree() -> None:
    manifest = _load_pixi_manifest()
    assert manifest["tasks"]["package-smoke"] == "python scripts/check_built_distribution.py"

    script = (_repo_root() / "scripts" / "check_built_distribution.py").read_text(
        encoding="utf-8"
    )
    for required_fragment in [
        "sabench-*.whl",
        "sabench-*.tar.gz",
        "--force-reinstall",
        "--no-deps",
        "cwd=tmpdir",
        "env=_smoke_environment()",
        "source_package_root = repo_root / \"sabench\"",
        "is_relative_to(source_package_root)",
        "get_transform(\"affine_a2_b1\")",
        "apply_transform(values, \"affine_a2_b1\")",
    ]:
        assert required_fragment in script


def test_ci_workflow_uses_pixi_ci_environment_and_named_tasks() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "prefix-dev/setup-pixi" in workflow
    assert "locked: true" in workflow
    assert "environments: ci" in workflow
    for command in [
        "pixi run -e ci lint",
        "pixi run -e ci fmt-check",
        "pixi run -e ci typecheck",
        "pixi run -e ci test-cov",
        "pixi run -e ci build",
        "pixi run -e ci twine-check",
        "pixi run -e ci package-smoke",
    ]:
        assert command in workflow
    assert 'pip install ".[dev]"' not in workflow
    assert "sabench/tests" not in workflow


def test_ci_workflow_avoids_legacy_raw_toolchain_commands() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for legacy_command in [
        "pip install .[dev]",
        "pip install build twine",
        "pixi run -e ci python -m build",
        "pixi run -e ci twine check --strict dist/*",
    ]:
        assert legacy_command not in workflow


def test_git_index_does_not_track_generated_or_handoff_artifacts() -> None:
    repo_root = _repo_root()
    if not (repo_root / ".git").exists():
        pytest.skip("tracked-file hygiene requires a git checkout")

    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_paths = sorted(result.stdout.splitlines())

    forbidden_exact_paths = {
        ".coverage",
        "coverage.xml",
        "REMOVE_FILES.txt",
        "REMOVED_PATHS.txt",
    }
    forbidden_dir_names = {
        "__MACOSX",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".pixi",
        "dist",
        "build",
        "htmlcov",
    }
    forbidden_suffixes = (".pyc", ".pyo", ".whl", ".zip", ".tar.gz")

    forbidden_tracked_paths = []
    for tracked_path in tracked_paths:
        path_parts = PurePosixPath(tracked_path).parts
        name = path_parts[-1]
        if tracked_path in forbidden_exact_paths:
            forbidden_tracked_paths.append(tracked_path)
        elif any(part in forbidden_dir_names or part.endswith(".egg-info") for part in path_parts):
            forbidden_tracked_paths.append(tracked_path)
        elif name == ".DS_Store" or name.startswith("._") or name == "diff.txt":
            forbidden_tracked_paths.append(tracked_path)
        elif tracked_path.startswith(".coverage") or tracked_path.endswith(forbidden_suffixes):
            forbidden_tracked_paths.append(tracked_path)

    assert forbidden_tracked_paths == []


def test_gitignore_blocks_local_bundle_and_platform_artifacts() -> None:
    gitignore = (_repo_root() / ".gitignore").read_text(encoding="utf-8")
    for pattern in [
        ".DS_Store",
        "__MACOSX/",
        "._*",
        "*.zip",
        "*.tar.gz",
        "coverage.xml",
        "diff.txt",
    ]:
        assert pattern in gitignore


def test_clean_stage_removes_build_artifacts_recursively() -> None:
    clean_section = _clean_section()

    assert (
        'find . -type d \\( -name "dist" -o -name "build" -o -name "*.egg-info" \\)'
        in clean_section
    )
    assert '-not -path "./.git/*" -not -path "./.pixi/*"' in clean_section
    assert '-prune -exec rm -rf {} +' in clean_section

    for root_only_command in [
        "rm -rf dist/ build/ sabench.egg-info/ src/*.egg-info",
        "rm -rf dist/ build/ sabench.egg-info/",
    ]:
        assert root_only_command not in clean_section


def test_clean_stage_removes_local_bundle_and_platform_artifacts_recursively() -> None:
    clean_section = _clean_section()

    assert 'find . -type d -name "__MACOSX"' in clean_section
    assert '-not -path "./.git/*" -not -path "./.pixi/*"' in clean_section
    assert '-prune -exec rm -rf {} +' in clean_section

    assert "find . -type f \\(" in clean_section
    for pattern in [
        '-name ".DS_Store"',
        '-name "._*"',
        '-name "diff.txt"',
        '-name "*.zip"',
        '-name "*.tar.gz"',
    ]:
        assert pattern in clean_section
    assert "-delete" in clean_section

    for root_only_command in [
        "rm -rf __MACOSX",
        "rm -f ._.* diff.txt *.zip *.tar.gz",
    ]:
        assert root_only_command not in clean_section
