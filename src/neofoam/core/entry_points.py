# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Entry-point groups through which a separately installed package extends neofoam.

Use it when the extension lives in its own repository (e.g. ``neofoam_mdao``);
code inside neofoam registers directly with :class:`~neofoam.core.plugin_system.PluginSystem`.

Two groups are read:

* ``neofoam.plugins`` — each entry point names a **module**. Importing it runs its
  registrations (``Model(...).register_with(viscosityModel)``, ``@Node.register``,
  …), so a plugin can add to *any* PluginSystem family. :func:`load_plugins`
  imports them all; every ``neofoam`` process entry point calls it before building
  models or schemas.
* ``neofoam.solvers`` — each entry point names a ``run(argv: list[str]) -> None``
  callable, exposed as ``neofoam solver <lower-cased entry point name>``.

A plugin that fails to import raises: a half-loaded extension would silently
change which models a case resolves to.

Example ``pyproject.toml`` of a plugin package::

    [project.entry-points."neofoam.plugins"]
    mdao = "neofoam_mdao.models"

    [project.entry-points."neofoam.solvers"]
    mdaofluid = "neofoam_mdao.solver:run"
"""

from importlib import metadata

__all__ = ["PLUGIN_GROUP", "SOLVER_GROUP", "load_plugins", "solver_entry_points"]

PLUGIN_GROUP = "neofoam.plugins"
SOLVER_GROUP = "neofoam.solvers"


def load_plugins() -> None:
    """Import every installed ``neofoam.plugins`` module.

    Call it at a process entry point, before models or schemas are built. Code inside
    neofoam registers directly with :class:`~neofoam.core.plugin_system.PluginSystem`
    and does not need it.

    Repeat calls are cheap and run no module body twice, because Python's import cache
    holds each module; a module that failed to import is not cached, so a retry raises
    again. Concurrent calls are safe for the same reason: the import system locks each
    module while it runs.

    Example::

        load_plugins()
        spec = resolve_solver("incompressibleFluid")
    """
    for entry_point in metadata.entry_points(group=PLUGIN_GROUP):
        try:
            entry_point.load()
        except Exception as exc:
            raise RuntimeError(
                f"neofoam plugin {entry_point.name!r} ({entry_point.value}) failed to load: {exc}"
            ) from exc


def solver_entry_points() -> list[metadata.EntryPoint]:
    """Return the installed ``neofoam.solvers`` entry points without loading them.

    Use it to find plugin solver commands; use :func:`load_plugins` for modules that
    register models.

    Imports no plugin code, so it is safe to call before a command is chosen.

    Example::

        names = [entry_point.name for entry_point in solver_entry_points()]
    """
    return list(metadata.entry_points(group=SOLVER_GROUP))
