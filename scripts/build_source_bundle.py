from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

EXCLUDED_DIR_NAMES = {
    ".git",
    "__MACOSX",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".pixi",
    "dist",
    "build",
}

EXCLUDED_FILE_NAMES = {
    ".DS_Store",
    "coverage.xml",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".whl",
    ".zip",
}

EXCLUDED_SUFFIX_CHAINS = {
    (".tar", ".gz"),
}


def _has_excluded_suffix_chain(path: Path) -> bool:
    suffixes = tuple(path.suffixes)
    return any(suffixes[-len(chain) :] == chain for chain in EXCLUDED_SUFFIX_CHAINS)


def is_excluded(relative_path: Path) -> bool:
    parts = relative_path.parts
    if any(part in EXCLUDED_DIR_NAMES for part in parts[:-1]):
        return True

    name = relative_path.name
    if name in EXCLUDED_DIR_NAMES or name in EXCLUDED_FILE_NAMES:
        return True
    if name.startswith(".coverage"):
        return True
    if any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return True
    if _has_excluded_suffix_chain(relative_path):
        return True
    return False


def _iter_candidate_files(source_root: Path, selected_paths: list[Path] | None) -> list[Path]:
    if not selected_paths:
        candidates = [path for path in source_root.rglob("*") if path.is_file()]
    else:
        candidates = []
        for relative_path in selected_paths:
            candidate = source_root / relative_path
            if not candidate.exists():
                raise FileNotFoundError(f"Selected path does not exist: {relative_path}")
            if candidate.is_dir():
                candidates.extend(path for path in candidate.rglob("*") if path.is_file())
            else:
                candidates.append(candidate)

    unique_candidates = sorted({path.resolve() for path in candidates})
    return [Path(path) for path in unique_candidates]


def _zipinfo_for_path(source_path: Path, archive_name: str) -> ZipInfo:
    zipinfo = ZipInfo.from_file(source_path, arcname=archive_name)
    zipinfo.compress_type = ZIP_DEFLATED
    mode = stat.S_IMODE(source_path.stat().st_mode)
    zipinfo.external_attr = (mode & 0xFFFF) << 16
    return zipinfo


def build_source_bundle(source_root: Path, output_path: Path, selected_paths: list[Path] | None = None) -> None:
    source_root = source_root.resolve()
    output_path = output_path.resolve()

    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_files = _iter_candidate_files(source_root, selected_paths)

    with ZipFile(output_path, mode="w") as zf:
        for source_path in candidate_files:
            relative_path = source_path.relative_to(source_root)
            if is_excluded(relative_path):
                continue
            archive_name = relative_path.as_posix()
            with source_path.open("rb") as handle:
                zf.writestr(_zipinfo_for_path(source_path, archive_name), handle.read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean sabench source bundle.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--path",
        dest="paths",
        action="append",
        type=Path,
        default=None,
        help="Repo-relative file or directory to include. Repeat to make a changed-file bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_source_bundle(args.source_root, args.output, args.paths)


if __name__ == "__main__":
    main()
