"""Architecture contract tests for canonical transform registry ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from sabench.transforms import apply_transform, catalog, get_transform, registry

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_SOURCE = REPO_ROOT / "sabench" / "transforms" / "registry.py"
EVALUATION_SOURCE = REPO_ROOT / "sabench" / "transforms" / "evaluation.py"
LEGACY_TRANSFORMS_SOURCE = REPO_ROOT / "sabench" / "transforms" / "transforms.py"
PACKAGE_INIT_SOURCE = REPO_ROOT / "sabench" / "transforms" / "__init__.py"
TRANSFORMS_PACKAGE = REPO_ROOT / "sabench" / "transforms"

CATALOG_EXPORT_NAMES = {
    "TRANSFORMS",
    "LINEAR_TRANSFORMS",
    "POINTWISE_TRANSFORMS",
    "AFFINE_TRANSFORMS",
    "NONLOCAL_TRANSFORMS",
    "CONVEX_TRANSFORMS",
    "CONCAVE_TRANSFORMS",
    "MONOTONE_TRANSFORMS",
    "NONMONOTONE_TRANSFORMS",
    "SMOOTH_TRANSFORMS",
    "NONSMOOTH_TRANSFORMS",
}


def _assigned_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
    return names


def test_registry_imports_canonical_catalog_not_legacy_monolith() -> None:
    source = REGISTRY_SOURCE.read_text()
    assert "sabench.transforms.catalog" in source
    assert "sabench.transforms.transforms" not in source


def test_legacy_transform_monolith_module_is_removed() -> None:
    assert not LEGACY_TRANSFORMS_SOURCE.exists()


def test_evaluation_module_owns_public_apply_and_score_helpers() -> None:
    tree = ast.parse(EVALUATION_SOURCE.read_text())
    assigned = _assigned_names(tree)
    imported = _imported_names(tree)
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert CATALOG_EXPORT_NAMES.isdisjoint(assigned)
    assert not any(name.startswith("t_") for name in imported)
    assert function_names == {"_variance_weighted_s1", "apply_transform", "score_transform"}


def test_package_public_catalog_exports_come_from_registry_and_helpers_from_evaluation() -> None:
    tree = ast.parse(PACKAGE_INIT_SOURCE.read_text())
    registry_imports: set[str] = set()
    evaluation_imports: set[str] = set()
    legacy_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "sabench.transforms.registry":
            registry_imports.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module == "sabench.transforms.evaluation":
            evaluation_imports.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module == "sabench.transforms.transforms":
            legacy_imports.update(alias.name for alias in node.names)

    assert "TRANSFORMS" in registry_imports
    assert {"apply_transform", "score_transform"} <= evaluation_imports
    assert not legacy_imports


def test_transforms_package_has_no_legacy_monolith_imports() -> None:
    sources = sorted(
        path for path in TRANSFORMS_PACKAGE.glob("*.py") if path.name != "transforms.py"
    )
    offenders = [path for path in sources if "sabench.transforms.transforms" in path.read_text()]
    assert offenders == []


def test_registry_uses_catalog_mapping_identity() -> None:
    assert registry.TRANSFORMS is catalog.TRANSFORMS
    assert set(registry.TRANSFORMS) == set(registry.TRANSFORM_REGISTRY)


def test_apply_transform_delegates_to_typed_registry() -> None:
    y = np.array([-1.0, 0.0, 2.0])
    np.testing.assert_allclose(apply_transform(y, "affine_a2_b1"), get_transform("affine_a2_b1")(y))
