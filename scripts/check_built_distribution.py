#!/usr/bin/env python
"""Smoke-test built sabench distribution artifacts.

This script is intended to run after ``python -m build`` and ``twine check``.
It verifies that the build produced exactly one wheel and one source archive,
installs the wheel into the current environment without resolving dependencies,
and imports the installed package from outside the source package tree so the
smoke test cannot be satisfied by ``sabench/`` in the working tree.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _single_artifact(dist_dir: Path, pattern: str) -> Path:
    """Return the single artifact matching ``pattern`` in ``dist_dir``."""
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        formatted = ", ".join(path.name for path in matches) or "<none>"
        raise RuntimeError(
            f"Expected exactly one artifact matching {pattern!r} in {dist_dir}; "
            f"found {len(matches)}: {formatted}"
        )
    return matches[0]


def _smoke_environment() -> dict[str, str]:
    """Return a subprocess environment that does not import from local paths."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _install_wheel(wheel_path: Path) -> None:
    """Install the built wheel into the current Python environment."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(wheel_path),
        ],
        check=True,
    )


def _run_import_smoke(repo_root: Path) -> None:
    """Import the installed wheel from outside the source package tree."""
    smoke_code = r'''
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np

import sabench
from sabench.benchmarks import BENCHMARK_REGISTRY, list_benchmarks
from sabench.transforms import apply_transform, get_transform, list_transforms

repo_root = Path(__REPO_ROOT__).resolve()
source_package_root = repo_root / "sabench"
package_file = Path(sabench.__file__).resolve()
if package_file.is_relative_to(source_package_root):
    raise SystemExit(f"Imported sabench from source package tree: {package_file}")

if version("sabench") != sabench.__version__:
    raise SystemExit(
        f"Distribution metadata version {version('sabench')} does not match "
        f"sabench.__version__ {sabench.__version__}"
    )

package_root = package_file.parent
if (package_root / "tests").exists():
    raise SystemExit(f"Runtime package contains tests directory: {package_root / 'tests'}")
if (package_root / "transforms" / "transforms.py").exists():
    raise SystemExit("Runtime package contains removed transform monolith")

benchmark_names = list_benchmarks()
transform_names = list_transforms()
if not benchmark_names or not BENCHMARK_REGISTRY:
    raise SystemExit("Installed package exposes no benchmarks")
if not transform_names:
    raise SystemExit("Installed package exposes no transforms")

values = np.array([1.0, 2.0, 3.0])
affine_transform = get_transform("affine_a2_b1")
transformed = apply_transform(values, "affine_a2_b1")
if not np.allclose(transformed, affine_transform(values)):
    raise SystemExit("Installed transform smoke check failed")

print(f"sabench wheel smoke OK: {package_file}")
'''.replace("__REPO_ROOT__", repr(str(repo_root)))

    with tempfile.TemporaryDirectory(prefix="sabench-wheel-smoke-") as tmpdir:
        subprocess.run(
            [sys.executable, "-c", smoke_code],
            cwd=tmpdir,
            env=_smoke_environment(),
            check=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="Directory containing artifacts produced by python -m build.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    dist_dir = args.dist_dir.resolve()
    if not dist_dir.is_dir():
        raise RuntimeError(f"Distribution directory does not exist: {dist_dir}")

    wheel_path = _single_artifact(dist_dir, "sabench-*.whl")
    sdist_path = _single_artifact(dist_dir, "sabench-*.tar.gz")

    print(f"Found wheel: {wheel_path.name}")
    print(f"Found sdist: {sdist_path.name}")
    _install_wheel(wheel_path)
    _run_import_smoke(repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
