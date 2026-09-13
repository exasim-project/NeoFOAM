# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The script front door: execute a case's ``system/postProcess.py``.

Executing case-local Python is the same trust boundary as running the case at
all, so the path is confined with a
:class:`~neofoam.tooling.workspace.Workspace` and the loader is called from
:func:`~neofoam.postprocess.table.tables_for_case` only — the one place a
solver run assembles a case's tables, and not part of the package's public
surface, so no MCP, agent or tooling frontend executes a case script by
importing it::

    tables = load_script(case_dir)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.postprocess.table import TableSet
from neofoam.tooling.workspace import Workspace

#: Where a case keeps its post-processing script.
SCRIPT_FILE = "system/postProcess.py"

_MODULE_NAME = "neofoam_case_post"

#: The only families a re-load may take registrations back from: a script's own
#: nodes, sources and writers. Anything it registers elsewhere (a write policy, a
#: turbulence model) is the process's, not the script's.
_SCRIPT_FAMILIES = ("Source", "Node", "TableWriter")

# What the scripts loaded so far registered with those families, so a re-load
# can take those registrations back (see :func:`_forget_script_plugins`).
_SCRIPT_PLUGINS: list[tuple[str, Type[BaseModel]]] = []


def load_script(case_dir: Path, rel_path: str = SCRIPT_FILE) -> Optional[TableSet]:
    """Execute a case's post-processing script and return its single TableSet.

    ``None`` when the case has no script. Run *before* the spec file is
    resolved, so the ``@Node.register`` classes the script defines are already
    part of the union the spec resolves against. Raises
    :class:`~neofoam.tooling.workspace.CaseAccessError` for a path that leaves
    the case and ``ValueError`` when the script does not define exactly one
    module-level :class:`~neofoam.postprocess.table.TableSet`.
    """
    if not (Path(case_dir) / rel_path).exists():
        return None
    path = Workspace.at(case_dir).resolve_existing(
        rel_path, require_dir=False, kind="post-process script"
    )

    _forget_script_plugins()
    before = _plugin_snapshot()
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"postProcess: {path} cannot be imported as a Python module")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: pydantic resolves a class defined in the
    # script through ``sys.modules``.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        # Also on failure: a script that raises *after* a ``@Node.register`` has
        # still registered that class, and leaving it unrecorded would leave the
        # family's union permanently ambiguous for the next load.
        _remember_script_plugins(before)

    found = [obj for obj in vars(module).values() if isinstance(obj, TableSet)]
    if len(found) != 1:
        raise ValueError(
            f"postProcess: {path} must define exactly one module-level TableSet, found {len(found)}"
        )
    return found[0]


def _plugin_snapshot() -> dict[str, list[Type[BaseModel]]]:
    """The classes the script-owned plugin families hold right now."""
    plugins = PluginSystem.list_plugins()
    return {family: list(plugins.get(family, [])) for family in _SCRIPT_FAMILIES}


def _remember_script_plugins(before: dict[str, list[Type[BaseModel]]]) -> None:
    """Record the source/node classes the script itself defined and registered.

    A class that appeared because the script *imported* something belongs to the
    module it was defined in and stays registered — only the script's own
    definitions (``__module__`` is the script module) are the script's to undo.
    """
    plugins = PluginSystem.list_plugins()
    for family in _SCRIPT_FAMILIES:
        known = before.get(family, [])
        _SCRIPT_PLUGINS.extend(
            (family, cls)
            for cls in plugins.get(family, [])
            if cls not in known and cls.__module__ == _MODULE_NAME
        )


def _forget_script_plugins() -> None:
    """Unregister what earlier script loads registered.

    Re-executing a script builds *new* classes for the same ``type`` literal,
    and two union members sharing a discriminator value make pydantic reject the
    union — while the class is being created, i.e. inside the script. So the
    older generation goes before the script runs again (last registration wins).
    """
    for family, plugin_cls in _SCRIPT_PLUGINS:
        PluginSystem.remove_plugin_model(family, plugin_cls)
    _SCRIPT_PLUGINS.clear()
