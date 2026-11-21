# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-name registry — filesystem-discovered, not hand-maintained.

Domain-level solver discovery: a solver is any package ``neofoam.solver.<name>``
carrying a module ``<name>.py`` that exports a ``<name>`` spec object (the convention
every solver package follows, e.g. ``incompressibleFluid/incompressibleFluid.py`` →
``incompressibleFluid``). New solvers register automatically by following that layout;
there is no per-solver line to add here.

Importing a solver spec pulls ``pybFoam`` (the native bindings), so discovery is
**filesystem-only** (no spec import) and :func:`resolve_solver` imports the spec lazily
(pulling ``pybFoam``) only when actually called. This lives in ``framework`` rather than
in a frontend package so both the MCP host (``neofoam.mcp.registry`` re-exports it) and
the tooling/sweep runners depend on the framework, not on each other.

Separately installed packages add solvers through the ``neofoam.region_solvers`` entry
point group (:mod:`neofoam.core.entry_points`). A built-in solver of the same name wins.
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
    """Region solvers of installed packages by name, unloaded."""
    return {entry_point.name: entry_point for entry_point in region_solver_entry_points()}


def list_solver_names() -> list[str]:
    """Registered solver names, built-in and from plugins; loads no spec (no pybFoam)."""
    return sorted({*_discover_solver_names(), *_plugin_solvers()})


def resolve_solver(name: str) -> Any:
    """Resolve a registered solver spec by name; built-in names win, unknown names fail fast."""
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
