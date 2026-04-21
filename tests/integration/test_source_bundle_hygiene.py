from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

SCRIPT = Path("scripts/build_source_bundle.py")


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
    (repo_root / "coverage.xml").write_text("<coverage/>\n", encoding="utf-8")
    (repo_root / "dist" / "artifact.whl").write_bytes(b"wheel")
    (repo_root / "result.zip").write_bytes(b"zip")

    output_path = tmp_path / "bundle.zip"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(repo_root),
            "--output",
            str(output_path),
        ],
        check=True,
    )

    with ZipFile(output_path) as zf:
        names = set(zf.namelist())
        assert "sabench/__init__.py" in names
        assert "tests/integration/test_ok.py" in names
        assert "scripts/example.sh" in names

        assert ".git/config" not in names
        assert "__MACOSX/._junk" not in names
        assert "__pycache__/junk.pyc" not in names
        assert ".pytest_cache/state" not in names
        assert ".coverage" not in names
        assert "coverage.xml" not in names
        assert "dist/artifact.whl" not in names
        assert "result.zip" not in names

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

    output_path = tmp_path / "changed.zip"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(repo_root),
            "--output",
            str(output_path),
            "--path",
            "scripts/keep.py",
            "--path",
            "tests/integration/test_keep.py",
            "--path",
            "coverage.xml",
        ],
        check=True,
    )

    with ZipFile(output_path) as zf:
        names = set(zf.namelist())
        assert names == {"scripts/keep.py", "tests/integration/test_keep.py"}
