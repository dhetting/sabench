"""Release-facing API and documentation consistency tests."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import sabench
from sabench.benchmarks import BENCHMARK_REGISTRY
from sabench.transforms import (
    CONCAVE_TRANSFORMS,
    CONVEX_TRANSFORMS,
    MONOTONE_TRANSFORMS,
    NONLOCAL_TRANSFORMS,
    NONSMOOTH_TRANSFORMS,
    POINTWISE_TRANSFORMS,
    SMOOTH_TRANSFORMS,
    TRANSFORMS,
    list_transforms,
)


def _repo_root() -> Path:
    return Path(sabench.__file__).resolve().parent.parent


def _readme_text() -> str:
    return (_repo_root() / "README.md").read_text(encoding="utf-8")


def _demo_notebook() -> dict[str, object]:
    notebook_path = _repo_root() / "notebooks" / "demo.ipynb"
    return json.loads(notebook_path.read_text(encoding="utf-8"))


def _demo_code_cells() -> str:
    notebook = _demo_notebook()
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _assigned_tuple_keys(source: str, variable_name: str) -> set[str]:
    module = ast.parse(source)
    keys: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        assigned_to_variable = any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        )
        if not assigned_to_variable:
            continue
        if not isinstance(node.value, ast.List):
            continue
        for element in node.value.elts:
            if (
                isinstance(element, ast.Tuple)
                and element.elts
                and isinstance(element.elts[0], ast.Constant)
                and isinstance(element.elts[0].value, str)
            ):
                keys.add(element.elts[0].value)
    return keys


def test_root_public_api_is_minimal() -> None:
    assert sabench.__all__ == ["__version__"]
    assert not hasattr(sabench, "Ishigami")
    assert not hasattr(sabench, "saltelli_sample")
    assert not hasattr(sabench, "apply_transform")
    assert not hasattr(sabench, "TRANSFORMS")


def test_readme_uses_subpackage_imports() -> None:
    readme = _readme_text()
    assert "from sabench import" not in readme
    assert "from sabench.benchmarks.scalar import Ishigami" in readme
    assert "from sabench.transforms import get_transform" in readme


def test_demo_notebook_uses_subpackage_imports() -> None:
    notebook = (_repo_root() / "notebooks" / "demo.ipynb").read_text(encoding="utf-8")
    assert "from sabench import" not in notebook
    assert "from sabench.benchmarks.scalar import Ishigami" in notebook
    assert "from sabench.transforms import" in notebook


def test_readme_benchmark_counts_match_registry() -> None:
    readme = _readme_text()
    family_counts = Counter(
        definition.spec.family for definition in BENCHMARK_REGISTRY.values()
    )

    assert f"**{family_counts['scalar']} scalar benchmarks**" in readme
    assert f"**{family_counts['functional']} functional / PDE benchmarks**" in readme
    assert f"**{family_counts['spatial']} spatial benchmarks**" in readme


def test_readme_transform_counts_match_catalog_metadata() -> None:
    readme = _readme_text()
    metadata_path = _repo_root() / "sabench" / "metadata" / "transforms_metadata.json"
    transform_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    category_counts = Counter(
        metadata["category"] for metadata in transform_metadata.values()
    )

    assert f"**{len(TRANSFORMS)} output transformations**" in readme
    assert f"## Transform catalogue ({len(TRANSFORMS)} transforms)" in readme

    for category, count in category_counts.items():
        display_category = category.title()
        table_pattern = rf"\| {re.escape(display_category)} \| {count} \|"
        assert re.search(table_pattern, readme), display_category


def test_readme_property_set_counts_match_registry() -> None:
    readme = _readme_text()

    expected_property_snippets = [
        f"POINTWISE_TRANSFORMS,   # {len(POINTWISE_TRANSFORMS)} element-wise transforms",
        f"NONLOCAL_TRANSFORMS,    # {len(NONLOCAL_TRANSFORMS)} transforms",
        f"CONVEX_TRANSFORMS,      # {len(CONVEX_TRANSFORMS)} transforms",
        f"CONCAVE_TRANSFORMS,     # {len(CONCAVE_TRANSFORMS)} transforms",
        f"MONOTONE_TRANSFORMS,    # {len(MONOTONE_TRANSFORMS)} monotone transforms",
        f"SMOOTH_TRANSFORMS,      # {len(SMOOTH_TRANSFORMS)} C² or smoother transforms",
        f"NONSMOOTH_TRANSFORMS,   # {len(NONSMOOTH_TRANSFORMS)} C⁰ or discontinuous transforms",
    ]
    for snippet in expected_property_snippets:
        assert snippet in readme


def test_demo_notebook_transform_keys_are_registered() -> None:
    registered_keys = set(list_transforms())
    source = _demo_code_cells()

    notebook_keys = set()
    transform_list_names = ["scalar_transforms", "temporal_transforms", "spatial_transforms"]
    for variable_name in transform_list_names:
        notebook_keys.update(_assigned_tuple_keys(source, variable_name))

    assert notebook_keys
    assert notebook_keys <= registered_keys


def test_demo_notebook_catalog_counts_match_registry() -> None:
    notebook_text = (_repo_root() / "notebooks" / "demo.ipynb").read_text(encoding="utf-8")

    assert "len(TRANSFORMS)" in notebook_text
    assert "transforms registered" in notebook_text
    assert "Transform catalogue overview" in notebook_text
    assert f"Pointwise ({len(POINTWISE_TRANSFORMS)})" in notebook_text
    assert f"Monotone ({len(MONOTONE_TRANSFORMS)})" in notebook_text
    assert f"Concave ({len(CONCAVE_TRANSFORMS)})" in notebook_text
    assert f"Convex ({len(CONVEX_TRANSFORMS)})" in notebook_text
    assert f"Smooth C-inf ({len(SMOOTH_TRANSFORMS)})" in notebook_text
    assert f"{len(TRANSFORMS)} output transform catalogue" in notebook_text
