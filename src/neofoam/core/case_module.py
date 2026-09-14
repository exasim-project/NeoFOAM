# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Execute a case-local Python module, and take back what it registered.

The mechanics shared by the case's script front doors
(:mod:`neofoam.postprocess.script`, :mod:`neofoam.preprocess.script`): the path
is confined with a :class:`~neofoam.tooling.workspace.Workspace`, the module is
executed under a private name, and the plugin classes the script itself defined
are recorded so the next load of the *same* script can unregister them. One
:class:`CaseModuleLoader` per script kind, created at module level::

    loader = CaseModuleLoader(module_name="neofoam_case_post",
                              families=("Node",), kind="postProcess")
    module = loader.load(case_dir, "system/postProcess.py")
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.tooling.workspace import Workspace

T = TypeVar("T")


class CaseModuleLoader:
    """Executes one kind of case script and owns the registrations it leaves behind.

    Executing case-local Python is the same trust boundary as running the case at
    all, so a loader is created by the module that owns that front door and is
    never part of a package's public surface — no MCP, agent or tooling frontend
    executes a case script by importing one. ``families`` names the plugin
    families a re-load may take registrations back from: a script's own classes,
    not a write policy or a turbulence model it happens to pull in. Not thread
    safe — the loader mutates the process-wide module table and plugin
    registries::

        _LOADER = CaseModuleLoader(module_name="neofoam_case_set_fields",
                                   families=("Node",), kind="setFields")
    """

    def __init__(self, *, module_name: str, families: tuple[str, ...], kind: str) -> None:
        self._module_name = module_name
        self._families = families
        self._kind = kind
        # What the scripts loaded so far registered with those families, so a
        # re-load can take those registrations back (see :meth:`_forget`).
        self._registered: list[tuple[str, Type[BaseModel]]] = []

    def load(self, case_dir: Path, rel_path: str) -> Optional[ModuleType]:
        """Execute ``case_dir/rel_path`` and return the module, or ``None`` if absent.

        Raises :class:`~neofoam.tooling.workspace.CaseAccessError` for a path that
        leaves the case.
        """
        if not (Path(case_dir) / rel_path).exists():
            return None
        path = Workspace.at(case_dir).resolve_existing(
            rel_path, require_dir=False, kind=f"{self._kind} script"
        )

        self._forget()
        before = self._snapshot()
        spec = importlib.util.spec_from_file_location(self._module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"{self._kind}: {path} cannot be imported as a Python module")
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
            self._remember(before)
        return module

    def _snapshot(self) -> dict[str, list[Type[BaseModel]]]:
        """The classes the script-owned plugin families hold right now."""
        plugins = PluginSystem.list_plugins()
        return {family: list(plugins.get(family, [])) for family in self._families}

    def _remember(self, before: dict[str, list[Type[BaseModel]]]) -> None:
        """Record the plugin classes the script itself defined and registered.

        A class that appeared because the script *imported* something belongs to the
        module it was defined in and stays registered — only the script's own
        definitions (``__module__`` is the script module) are the script's to undo.
        """
        plugins = PluginSystem.list_plugins()
        for family in self._families:
            known = before.get(family, [])
            self._registered.extend(
                (family, cls)
                for cls in plugins.get(family, [])
                if cls not in known and cls.__module__ == self._module_name
            )

    def _forget(self) -> None:
        """Unregister what earlier loads of this script kind registered.

        Re-executing a script builds *new* classes for the same ``type`` literal,
        and two union members sharing a discriminator value make pydantic reject the
        union — while the class is being created, i.e. inside the script. So the
        older generation goes before the script runs again (last registration wins).
        """
        for family, plugin_cls in self._registered:
            PluginSystem.remove_plugin_model(family, plugin_cls)
        self._registered.clear()


def single_instance(module: ModuleType, cls: Type[T], kind: str) -> T:
    """The module's one module-level instance of ``cls``; none or several is an error."""
    found = [obj for obj in vars(module).values() if isinstance(obj, cls)]
    if len(found) != 1:
        raise ValueError(
            f"{kind}: {module.__file__} must define exactly one module-level "
            f"{cls.__name__}, found {len(found)}"
        )
    return found[0]
