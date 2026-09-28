# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Two package invariants: the declaration layer is pure numpy, the loaders are private.

Keeping the simulation backend out of the two front doors is what makes them
unit-testable without a mesh and what keeps them usable from a frontend that must
not pull OpenFOAM in: a declaration is selectors and numbers, and
:mod:`neofoam.preprocess.apply` is the single module that reads or writes a field.
The modules of that layer are listed below and their source is read for a pybFoam
import.

The second invariant is the script execution boundary: the loaders behind the
case's front doors belong to the ``setFields`` tool, so re-exporting one from the
package — where an MCP or agent frontend would import it — is a regression.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import neofoam.preprocess

PACKAGE_DIR = Path(neofoam.preprocess.__file__).parent

#: The declaration layer: the two front doors.
NUMPY_ONLY_MODULES = sorted(PACKAGE_DIR / name for name in ("config.py", "script.py"))

#: The case loaders — importable from their modules, never from the package.
CASE_LOADERS = ("load_config", "load_script", "set_fields_for_case")


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
def test_no_declaration_layer_module_imports_pybfoam(module: Path) -> None:
    assert "pybFoam" not in _imported_roots(module.read_text())


def test_the_case_loaders_are_not_on_the_packages_public_surface() -> None:
    exported = [name for name in CASE_LOADERS if name in neofoam.preprocess.__all__]

    assert exported == []
