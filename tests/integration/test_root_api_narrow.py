from pathlib import Path

import sabench


def _repo_root() -> Path:
    return Path(sabench.__file__).resolve().parent.parent


def test_root_public_api_is_minimal() -> None:
    assert sabench.__all__ == ["__version__"]
    assert not hasattr(sabench, "Ishigami")
    assert not hasattr(sabench, "saltelli_sample")
    assert not hasattr(sabench, "apply_transform")
    assert not hasattr(sabench, "TRANSFORMS")


def test_readme_uses_subpackage_imports() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    assert "from sabench import" not in readme
    assert "from sabench.benchmarks.scalar import Ishigami" in readme
    assert "from sabench.transforms import get_transform" in readme


def test_demo_notebook_uses_subpackage_imports() -> None:
    notebook = (_repo_root() / "notebooks" / "demo.ipynb").read_text(encoding="utf-8")
    assert "from sabench import" not in notebook
    assert "from sabench.benchmarks.scalar import Ishigami" in notebook
    assert "from sabench.transforms import" in notebook
