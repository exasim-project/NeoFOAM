# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluid solver package.

Minimal port of the ``feat/python_solvers`` solver compatible with the
``stack/python_arch`` framework API (SolverSpec / ModelSpec /
StagedInitSpec). Only the PIMPLE pressure-velocity algorithm is wired up
in this port; SIMPLE / PISO / boussinesq / SA variants from the source
branch are intentionally omitted.
"""

from .config_schema import config_classes
from .incompressibleFluid import incompressibleFluid, run

__all__ = ["incompressibleFluid", "run", "config_classes"]
