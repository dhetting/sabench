from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pytest

SCRIPT = Path("scripts/build_source_bundle.py")


def _forbidden_source_bundle_paths(names: set[str]) -> list[str]:
    forbidden_exact_names = {
        ".DS_Store",
        "coverage.xml",
        "diff.txt",
    }
    forbidden_dir_names = {
        ".git",
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

    forbidden_paths = []
    for archive_name in sorted(names):
        path = PurePosixPath(archive_name)
        path_parts = path.parts
        name = path_parts[-1]
        if archive_name.startswith("/") or ".." in path_parts:
            forbidden_paths.append(archive_name)
        elif any(part in forbidden_dir_names or part.endswith(".egg-info") for part in path_parts):
            forbidden_paths.append(archive_name)
        elif name in forbidden_exact_names or name.startswith("._"):
            forbidden_paths.append(archive_name)
        elif archive_name.startswith(".coverage") or archive_name.endswith(forbidden_suffixes):
            forbidden_paths.append(archive_name)

    return forbidden_paths


def _build_source_bundle(
    source_root: Path, output_path: Path, paths: list[str] | None = None
) -> None:
    command = [
        sys.executable,
        str(SCRIPT),
        "--source-root",
        str(source_root),
        "--output",
        str(output_path),
    ]
    for path in paths or []:
        command.extend(["--path", path])

    subprocess.run(command, check=True)


def _bundle_names(output_path: Path) -> set[str]:
    with ZipFile(output_path) as zf:
        return set(zf.namelist())


def test_actual_repo_source_bundle_is_clean(tmp_path: Path) -> None:
    repo_root = Path.cwd()
    output_path = tmp_path / "sabench-source.zip"

    _build_source_bundle(repo_root, output_path)

    names = _bundle_names(output_path)
    assert "sabench/__init__.py" in names
    assert "scripts/build_source_bundle.py" in names
    assert "tests/integration/test_source_bundle_hygiene.py" in names
    assert _forbidden_source_bundle_paths(names) == []


def test_source_bundle_excludes_transient_repo_artifacts_and_preserves_executable_bits(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    (repo_root / "sabench").mkdir()
    (repo_root / "tests" / "integration").mkdir(parents=True)
    (repo_root / "scripts").mkdir()
    (repo_root / ".git").mkdir()
    (repo_root / "__MACOSX").mkdir()
    (repo_root / "__pycache__").mkdir()
    (repo_root / ".pytest_cache").mkdir()
    (repo_root / "dist").mkdir()
    (repo_root / "sabench.egg-info").mkdir()
    (repo_root / "nested" / "generated.egg-info").mkdir(parents=True)

    (repo_root / "sabench" / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (repo_root / "tests" / "integration" / "test_ok.py").write_text(
        "def test_ok():\n    pass\n", encoding="utf-8"
    )
    script_path = repo_root / "scripts" / "example.sh"
    script_path.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
    script_path.chmod(0o755)

    (repo_root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (repo_root / "__MACOSX" / "._junk").write_text("junk\n", encoding="utf-8")
    (repo_root / "__pycache__" / "junk.pyc").write_bytes(b"pyc")
    (repo_root / ".pytest_cache" / "state").write_text("cache\n", encoding="utf-8")
    (repo_root / ".coverage").write_text("coverage\n", encoding="utf-8")
    (repo_root / ".coverage.unit").write_text("coverage\n", encoding="utf-8")
    (repo_root / "coverage.xml").write_text("<coverage/>\n", encoding="utf-8")
    (repo_root / "diff.txt").write_text("temporary diff\n", encoding="utf-8")
    (repo_root / "._README.md").write_text("appledouble\n", encoding="utf-8")
    (repo_root / "dist" / "artifact.whl").write_bytes(b"wheel")
    (repo_root / "sabench.egg-info" / "PKG-INFO").write_text("metadata\n", encoding="utf-8")
    (repo_root / "nested" / "generated.egg-info" / "SOURCES.txt").write_text(
        "metadata\n", encoding="utf-8"
    )
    (repo_root / "result.zip").write_bytes(b"zip")
    (repo_root / "source.tar.gz").write_bytes(b"archive")

    output_path = tmp_path / "bundle.zip"
    _build_source_bundle(repo_root, output_path)

    with ZipFile(output_path) as zf:
        names = set(zf.namelist())
        assert "sabench/__init__.py" in names
        assert "tests/integration/test_ok.py" in names
        assert "scripts/example.sh" in names

        assert _forbidden_source_bundle_paths(names) == []

        script_info = zf.getinfo("scripts/example.sh")
        mode = (script_info.external_attr >> 16) & 0o777
        assert stat.S_IMODE(mode) & 0o111


def test_source_bundle_can_build_changed_file_bundle(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    (repo_root / "scripts").mkdir()
    (repo_root / "tests" / "integration").mkdir(parents=True)
    (repo_root / "scripts" / "keep.py").write_text("print('keep')\n", encoding="utf-8")
    (repo_root / "tests" / "integration" / "test_keep.py").write_text(
        "def test_keep():\n    assert True\n", encoding="utf-8"
    )
    (repo_root / "coverage.xml").write_text("<coverage/>\n", encoding="utf-8")
    (repo_root / "diff.txt").write_text("temporary diff\n", encoding="utf-8")
    (repo_root / "._keep.py").write_text("appledouble\n", encoding="utf-8")

    output_path = tmp_path / "changed.zip"
    _build_source_bundle(
        repo_root,
        output_path,
        paths=[
            "scripts/keep.py",
            "tests/integration/test_keep.py",
            "coverage.xml",
            "diff.txt",
            "._keep.py",
        ],
    )

    names = _bundle_names(output_path)
    assert names == {"scripts/keep.py", "tests/integration/test_keep.py"}
    assert _forbidden_source_bundle_paths(names) == []


def test_source_bundle_skips_symbolic_links(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    package_dir = repo_root / "sabench"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")

    external_file = tmp_path / "external_secret.py"
    external_file.write_text("SECRET = True\n", encoding="utf-8")
    symlink_path = package_dir / "external_secret.py"
    try:
        symlink_path.symlink_to(external_file)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    output_path = tmp_path / "bundle.zip"
    _build_source_bundle(repo_root, output_path)

    names = _bundle_names(output_path)
    assert "sabench/__init__.py" in names
    assert "sabench/external_secret.py" not in names
    assert _forbidden_source_bundle_paths(names) == []


def test_changed_file_bundle_rejects_selected_paths_outside_source_root(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("print('outside')\n", encoding="utf-8")

    for path_arg in ["../outside.py", str(outside_file)]:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--source-root",
                str(repo_root),
                "--output",
                str(tmp_path / "changed.zip"),
                "--path",
                path_arg,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "Selected path" in result.stderr
        assert "repo-relative" in result.stderr or "escapes source root" in result.stderr
