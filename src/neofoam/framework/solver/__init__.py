# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
neofoam.framework.solver — SolverSpec / SolverRuntime package.

Public API:
    SolverSpec    — immutable solver definition
    SolverRuntime — one instantiation of a spec, owns per-instance state
    Solver        — factory alias: Solver("Name") -> SolverSpec
    region_step   — a region's time-step body as one operation of a shared loop
    region_operations — a region's time-step operations for a driver's own graph
"""

from .configurations import Configurations, configurations
from .regions import region_operations, region_step
from .runtime import SolverRuntime, SolverState
from .spec import Solver, SolverSpec

__all__ = [
    "SolverSpec",
    "SolverRuntime",
    "SolverState",
    "Solver",
    "Configurations",
    "configurations",
    "region_step",
    "region_operations",
]
