# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluidNeoN solver package.

Framework port of the NeoN-backed ``neoPimpleFoam`` solver: the same
SolverSpec / ModelSpec / StagedInitSpec composition as ``incompressibleFluid``,
with the NeoN bindings (``neon._neon`` / ``neofoam.neofoam_bindings``) as the
backend. The PIMPLE and SIMPLE pressure-velocity algorithms are wired up.
"""

from .config_schema import config_classes
from .incompressibleFluidNeoN import incompressibleFluidNeoN, run

__all__ = ["incompressibleFluidNeoN", "run", "config_classes"]
