# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case configs of heatConductionNeoN: an OpenFOAM case, read through pybFoam.

The mesh (``constant/polyMesh``), ``0/T``, ``system/fvSchemes`` and
``system/fvSolution`` are read by the NeoN runtime; ``system/controlDict`` and
``constant/thermalProperties`` by these configs.
"""

from pydantic import Field

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.io import OF, BaseConfig, IOStrategy


@IOStrategy(OF("system/controlDict"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict`` — time stepping and write control."""

    application: str = "heatConductionNeoN"


@IOStrategy(OF("constant/thermalProperties"))
class ThermalPropertiesConfig(BaseConfig):
    """``constant/thermalProperties`` — constant solid properties (SI units).

    The diffusivity is ``alpha = k / (rho cp)``; the heat flux a coupled patch
    exchanges is ``k dT/dn``.

    Example::

        rho         7800;   // kg/m^3
        cp          500;    // J/(kg K)
        k           45;     // W/(m K)
        heatSource  0;      // W/m^3
    """

    rho: float = Field(gt=0)
    cp: float = Field(gt=0)
    k: float = Field(gt=0)
    heatSource: float = 0.0
