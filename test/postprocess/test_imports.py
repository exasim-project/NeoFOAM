# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Two package-wide invariants: no pybFoam in the value layer, and importing registers every node.

Keeping pybFoam out of the nodes is what makes them unit-testable without a mesh
and what lets one node serve a pybFoam and a NeoN field alike: a node sees a
dataset and the NeoN kernels, never a field library. pybFoam belongs to the
sources, the CSV writer and the reduce helper alone, so the modules of the value
layer are listed below and their source is read for a pybFoam import.

The second invariant is what makes a YAML ``type:`` string resolve: a node class
is registered by its module being imported, so the ``__init__.py`` chain (the
package's, and the ``nodes``/``sources``/``writers`` subpackages') must import
every one. It is checked in a subprocess because a pytest session that ran any other
module of this package has already imported the node modules directly, which
would make an in-process check pass even with the ``__init__`` import removed.
The lists below are compared against the registry itself, so a new node,
source or writer without a row here fails rather than going unchecked.

The third is the public surface: the loaders behind the case's front doors
belong to the ``postProcess`` model (R2) and are imported from their own
modules, so re-exporting one from the package is a regression.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import neofoam.postprocess

PACKAGE_DIR = Path(neofoam.postprocess.__file__).parent

#: The value layer: the data contract, the case-facing front doors, the writer
#: interface and every node.
NUMPY_ONLY_MODULES = sorted(
    [PACKAGE_DIR / name for name in ("node.py", "table.py", "config.py", "script.py", "model.py")]
    + [PACKAGE_DIR / "writers" / "writer.py"]
    + list((PACKAGE_DIR / "nodes").rglob("*.py"))
)


#: The case loaders and resolvers — importable from their modules, never from
#: the package.
CASE_LOADERS = ("load_config", "load_script", "resolve_table", "tables_for_case")


def _imported_roots(source: str) -> set[str]:
    """The top-level package of every ``import x`` / ``from x import y`` in a module."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_every_listed_module_exists() -> None:
    # the list is hand-written: a renamed module would drop out of the scan silently
    missing = [module for module in NUMPY_ONLY_MODULES if not module.is_file()]
    assert missing == []


@pytest.mark.parametrize(
    "module", NUMPY_ONLY_MODULES, ids=lambda path: str(path.relative_to(PACKAGE_DIR))
)
def test_no_value_layer_module_imports_pybfoam(module: Path) -> None:
    assert "pybFoam" not in _imported_roots(module.read_text())


def test_the_case_loaders_are_not_on_the_packages_public_surface() -> None:
    exported = [name for name in CASE_LOADERS if name in neofoam.postprocess.__all__]

    assert exported == []


# every node shipped with the package, with the smallest spec that validates
SHIPPED_NODES: list[dict[str, Any]] = [
    {"type": "box", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
    {"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": 1.0},
    {"type": "not", "region": {"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": 1.0}},
    {
        "type": "binary",
        "op": "and",
        "left": {"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": 1.0},
        "right": {"type": "sphere", "center": [1.0, 0.0, 0.0], "radius": 1.0},
    },
    {"type": "directional", "bins": [0.5], "direction": [1.0, 0.0, 0.0]},
    {"type": "mag"},
    {"type": "component", "index": 0},
    {"type": "area"},
    {"type": "sample", "field": "U"},
    {"type": "sum"},
    {"type": "mean"},
    {"type": "max"},
    {"type": "min"},
    {"type": "surfIntegrate"},
    {"type": "volIntegrate"},
    {"type": "rows"},
    {"type": "scale"},
    {"type": "print"},
]

# every source shipped with the package, with the smallest spec that validates
SHIPPED_SOURCES: list[dict[str, Any]] = [
    {"type": "internal", "field": "p"},
    {"type": "patch", "field": "p", "patch": "movingWall"},
    {
        "type": "line",
        "field": "U",
        "start": [0.0, 0.0, 0.0],
        "end": [0.0, 0.1, 0.0],
        "n_points": 5,
    },
    {"type": "plane", "point": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]},
    {"type": "isoSurface", "iso_field": "alpha.water", "iso_value": 0.5},
    {"type": "residuals"},
]

# every writer shipped with the package, with the smallest spec that validates
SHIPPED_WRITERS: list[dict[str, Any]] = [
    {"type": "csv"},
]

SHIPPED: dict[str, list[dict[str, Any]]] = {
    "Node": SHIPPED_NODES,
    "Source": SHIPPED_SOURCES,
    "TableWriter": SHIPPED_WRITERS,
}

_RESOLVE_IN_A_FRESH_INTERPRETER = """
import json
import sys

import neofoam.postprocess  # noqa: F401  # the only import: registration must come from here
from neofoam.core.plugin_system import PluginSystem

for family, specs in json.loads(sys.argv[1]).items():
    registry = PluginSystem.get_registered(family)
    for spec in specs:
        selected = registry.base_cls.create(**{registry.discriminator_variable: spec})
        plugin = getattr(selected, registry.discriminator_variable)
        assert plugin.type == spec["type"], f"{spec} resolved to {plugin.type}"
    registered = {cls.model_fields["type"].default for cls in registry.plugin_registry}
    tested = {spec["type"] for spec in specs}
    drift = f"untested {registered - tested}, stale {tested - registered}"
    assert registered == tested, f"{family}: {drift}"
"""


def test_importing_the_package_registers_every_shipped_plugin() -> None:
    resolved = subprocess.run(
        [sys.executable, "-c", _RESOLVE_IN_A_FRESH_INTERPRETER, json.dumps(SHIPPED)],
        capture_output=True,
        text=True,
    )

    assert resolved.returncode == 0, resolved.stderr
