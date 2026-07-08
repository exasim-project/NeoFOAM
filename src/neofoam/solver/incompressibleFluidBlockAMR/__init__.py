# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluidBlockAMR solver package.

A framework ``SolverSpec`` for a laminar incompressible Navier-Stokes solve on a
block-structured AMReX grid, backed by ``neon.blockamr``'s
``DSLIncompressibleSolver`` (MAC + nodal projection). Same SolverSpec / ModelSpec
/ StagedInitSpec composition as ``incompressibleFluid``; the projection engine
replaces pybFoam, and the outer time loop + field writer are reused verbatim from
the framework.
"""

from .config_schema import config_classes
from .incompressibleFluidBlockAMR import incompressibleFluidBlockAMR, run

__all__ = ["incompressibleFluidBlockAMR", "run", "config_classes"]
