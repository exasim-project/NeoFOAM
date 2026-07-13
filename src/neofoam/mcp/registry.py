# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-name registry.

Importing a solver spec pulls ``pybFoam`` (the native bindings), so the
registry must not import specs at module-import time. Loaders are lazy:
``list_solver_names()`` returns names only, and ``resolve_solver`` imports the
spec only when actually called.
"""

from __future__ import annotations

from typing import Any, Callable


def _load_incompressible_fluid() -> Any:
    from neofoam.solver.incompressibleFluid.incompressibleFluid import (
        incompressibleFluid,
    )

    return incompressibleFluid


def _load_incompressible_fluid_neon() -> Any:
    from neofoam.solver.incompressibleFluidNeoN.incompressibleFluidNeoN import (
        incompressibleFluidNeoN,
    )

    return incompressibleFluidNeoN


SOLVER_LOADERS: dict[str, Callable[[], Any]] = {
    "incompressibleFluid": _load_incompressible_fluid,
    "incompressibleFluidNeoN": _load_incompressible_fluid_neon,
}


def list_solver_names() -> list[str]:
    """Registered solver names (no spec import — pulls no pybFoam)."""
    return list(SOLVER_LOADERS)


def resolve_solver(name: str) -> Any:
    """Resolve a registered solver spec by name; unknown name fails fast."""
    try:
        loader = SOLVER_LOADERS[name]
    except KeyError as exc:
        known = ", ".join(SOLVER_LOADERS)
        raise ValueError(f"unknown solver {name!r}; known solvers: {known}") from exc
    return loader()
