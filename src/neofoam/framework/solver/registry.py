# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-name registry — discovered from the package layout and entry points.

Domain-level solver discovery: a solver is any package ``neofoam.solver.<name>``
carrying a module ``<name>.py`` that exports a ``<name>`` spec object (the convention
every solver package follows, e.g. ``incompressibleFluid/incompressibleFluid.py`` →
``incompressibleFluid``). New solvers register automatically by following that layout;
there is no per-solver line to add here.

Importing a solver spec pulls ``pybFoam`` (the native bindings), so discovery reads only
the package layout and the installed entry-point metadata (no spec import) and
:func:`resolve_solver` imports the spec lazily (pulling ``pybFoam``) only when actually
called. This lives in ``framework`` rather than
in a frontend package so both the MCP host (``neofoam.mcp.registry`` re-exports it) and
the tooling/sweep runners depend on the framework, not on each other.

Separately installed packages add solvers through the ``neofoam.region_solvers`` entry
point group (:mod:`neofoam.core.entry_points`). A built-in solver of the same name wins;
two entry points sharing another name with different targets raise ``ValueError``.

Example::

    spec = resolve_solver(list_solver_names()[0])
"""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
from typing import Any

import neofoam.solver  # empty package __init__ — imports no pybFoam
from neofoam.core.entry_points import region_solver_entry_points


def _discover_solver_names() -> list[str]:
    """Solver names from the package layout, sorted; imports no spec (no pybFoam)."""
    names: list[str] = []
    for pkg_root in neofoam.solver.__path__:
        root = Path(pkg_root)
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if (
                child.is_dir()
                and (child / "__init__.py").is_file()
                and (child / f"{child.name}.py").is_file()
            ):
                names.append(child.name)
    return names


def _plugin_solvers() -> dict[str, metadata.EntryPoint]:
    """Region solvers of installed packages by name, unloaded; conflicting targets raise."""
    built_in = set(_discover_solver_names())
    solvers: dict[str, metadata.EntryPoint] = {}
    for entry_point in region_solver_entry_points():
        if entry_point.name in built_in:
            continue
        first = solvers.setdefault(entry_point.name, entry_point)
        if first.value != entry_point.value:
            raise ValueError(
                f"region solver {entry_point.name!r} has conflicting targets: "
                f"{first.value!r} and {entry_point.value!r}"
            )
    return solvers


def list_solver_names() -> list[str]:
    """List the registered solver names, built-in and from installed packages.

    Use it to show or check solver names; use :func:`resolve_solver` to get a spec.

    Loads no spec. The names are sorted and unique. Two entry points that
    give a non-built-in name different targets raise ``ValueError``.

    Example::

        available = "incompressibleFluid" in list_solver_names()
    """
    return sorted({*_discover_solver_names(), *_plugin_solvers()})


def resolve_solver(name: str) -> Any:
    """Return the solver spec registered under ``name``.

    Use it when a case or tool selects a solver by name; use :func:`list_solver_names`
    to see the names without loading a spec.

    A built-in solver wins over an entry point of the same name and is resolved without
    reading entry points. For any other name, an unknown name or two entry points that
    give one name different targets raise ``ValueError``.

    Example::

        spec = resolve_solver("incompressibleFluid")
    """
    known = _discover_solver_names()
    if name in known:
        module = importlib.import_module(f"neofoam.solver.{name}.{name}")
        return getattr(module, name)
    plugins = _plugin_solvers()
    if name in plugins:
        return plugins[name].load()
    raise ValueError(
        f"unknown solver {name!r}; known solvers: {', '.join(sorted({*known, *plugins}))}"
    )
