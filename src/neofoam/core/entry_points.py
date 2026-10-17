# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Entry-point groups through which a separately installed package extends neofoam.

Use it when the extension lives in its own repository (e.g. ``neofoam_mdao``);
code inside neofoam registers directly with :class:`~neofoam.core.plugin_system.PluginSystem`.

Two groups are read:

* ``neofoam.plugins`` — each entry point names a **module**. Importing it runs its
  registrations (``Model(...).register_with(viscosityModel)``, ``@Node.register``,
  …), so a plugin can add to *any* PluginSystem family. :func:`load_plugins`
  imports them once per process; the ``neofoam`` CLI calls it before every command.
* ``neofoam.solvers`` — each entry point names a ``run(argv: list[str]) -> None``
  callable, exposed as ``neofoam solver <entry point name>``.

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

_loaded: set[str] = set()


def load_plugins() -> list[str]:
    """Import every ``neofoam.plugins`` module not imported yet and return their names."""
    newly_loaded: list[str] = []
    for entry_point in metadata.entry_points(group=PLUGIN_GROUP):
        if entry_point.name in _loaded:
            continue
        # Marked before the import so a plugin that itself calls load_plugins() cannot recurse.
        _loaded.add(entry_point.name)
        try:
            entry_point.load()
        except Exception as exc:
            raise RuntimeError(
                f"neofoam plugin {entry_point.name!r} ({entry_point.value}) failed to load: {exc}"
            ) from exc
        newly_loaded.append(entry_point.name)
    return newly_loaded


def solver_entry_points() -> list[metadata.EntryPoint]:
    """Return the ``neofoam.solvers`` entry points, unloaded."""
    return list(metadata.entry_points(group=SOLVER_GROUP))
