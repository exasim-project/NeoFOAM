# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-name registry — re-exported from the framework (back-compat shim).

The registry is a domain-level concern and now lives in
:mod:`neofoam.framework.solver.registry`; this module re-exports it so existing
``from neofoam.mcp.registry import resolve_solver`` imports keep working. New code
should import from the framework directly.
"""

from neofoam.framework.solver.registry import (
    list_solver_names,
    resolve_solver,
)

__all__ = ["list_solver_names", "resolve_solver"]
