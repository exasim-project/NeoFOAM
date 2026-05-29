# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
neofoam.framework.solver — SolverSpec / SolverRuntime package.

Public API:
    SolverSpec    — immutable solver definition
    SolverRuntime — one instantiation of a spec, owns per-instance state
    Solver        — factory alias: Solver("Name") -> SolverSpec
"""

from .spec import SolverSpec, Solver
from .runtime import SolverRuntime, SolverState
from .configurations import Configurations, configurations

__all__ = [
    "SolverSpec",
    "SolverRuntime",
    "SolverState",
    "Solver",
    "Configurations",
    "configurations",
]
