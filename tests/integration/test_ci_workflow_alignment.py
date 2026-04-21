from pathlib import Path


def _ci_workflow_text() -> str:
    return Path(".github/workflows/ci.yml").read_text(encoding="utf-8")


def test_ci_workflow_does_not_reference_removed_runtime_tests() -> None:
    workflow = _ci_workflow_text()

    assert "sabench/tests" not in workflow


def test_ci_workflow_uses_locked_pixi_environment() -> None:
    workflow = _ci_workflow_text()

    assert "prefix-dev/setup-pixi@" in workflow
    assert "locked: true" in workflow
    assert "pixi run -e ci" in workflow


def test_ci_workflow_build_job_uses_pixi_toolchain() -> None:
    workflow = _ci_workflow_text()

    assert "pixi run -e ci python -m build" in workflow
    assert "pixi run -e ci twine check --strict dist/*" in workflow
    assert "pip install build twine" not in workflow
