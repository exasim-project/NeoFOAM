# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Entry-point groups through which a separately installed package extends neofoam.

Use it when the extension lives in its own repository (e.g. ``neofoam_mdao``);
code inside neofoam registers directly with :class:`~neofoam.core.plugin_system.PluginSystem`.

Four groups extend neofoam; :mod:`neofoam.ui.plugins` reads the last, this module the rest:

* ``neofoam.plugins`` — each entry point names a **module**. Importing it runs its
  registrations (``Model(...).register_with(viscosityModel)``, ``@Node.register``,
  …), so a plugin can add to *any* PluginSystem family. :func:`load_plugins`
  imports them all; every ``neofoam`` process entry point calls it before building
  models or schemas.
  A module can also bind a new optional family with ``solver.models(family)``; the
  solver's load step detects it through ``SolverSpec.detect_optional_models``.
* ``neofoam.solvers`` — each entry point names a ``run(argv: list[str]) -> None``
  callable, exposed as ``neofoam solver <lower-cased entry point name>``.
* ``neofoam.region_solvers`` — each entry point names a ``SolverSpec``, which
  :func:`~neofoam.framework.solver.registry.resolve_solver` finds by name. A built-in
  solver of the same name takes precedence.
* ``neofoam.ui.steps`` — each entry point names a factory of an extra wizard step.

A plugin that fails to import raises: a half-loaded extension would silently
change which models a case resolves to.

Every installed ``neofoam.plugins`` module is imported when a neofoam process
starts, so install only plugin packages you trust.

Example ``pyproject.toml`` of a plugin package::

    [project.entry-points."neofoam.plugins"]
    mdao = "neofoam_mdao.models"

    [project.entry-points."neofoam.solvers"]
    mdaofluid = "neofoam_mdao.solver:run"

    [project.entry-points."neofoam.region_solvers"]
    mdaosolid = "neofoam_mdao.region:mdaosolid"
"""

from importlib import metadata

__all__ = [
    "PLUGIN_GROUP",
    "REGION_SOLVER_GROUP",
    "SOLVER_GROUP",
    "load_plugins",
    "region_solver_entry_points",
    "solver_entry_points",
]

PLUGIN_GROUP = "neofoam.plugins"
SOLVER_GROUP = "neofoam.solvers"
REGION_SOLVER_GROUP = "neofoam.region_solvers"


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


def region_solver_entry_points() -> list[metadata.EntryPoint]:
    """Return the installed ``neofoam.region_solvers`` entry points without loading them.

    Use it to inspect the raw entry points, including any a built-in solver shadows; use
    :func:`~neofoam.framework.solver.registry.resolve_solver` to get a spec by name.

    Imports no plugin code.

    Example::

        names = [entry_point.name for entry_point in region_solver_entry_points()]
    """
    return list(metadata.entry_points(group=REGION_SOLVER_GROUP))
