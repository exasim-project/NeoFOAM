# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case config of the multiRun solver: the shared time loop, the region solvers, their coupling."""

from typing import Literal

from pydantic import BaseModel, Field

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.io import YAML, BaseConfig, IOStrategy


@IOStrategy(YAML("system/controlDict.yaml"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict.yaml`` — the time loop every region shares, and its solvers.

    ``regionSolvers`` maps each region to the solver that runs it, in stepping
    order; each region keeps its own files under ``constant/<region>/``,
    ``0/<region>/`` and ``system/<region>/``. A region solver registers its
    time-step body with ``@<solver>.time_step``.

    Example::

        endTime: 10.0
        deltaT: 1.0
        regionSolvers:
          fluid: myFluidSolver
          solid: mySolidSolver
    """

    # The i18n prefix labels the wizard's "add a region" row (neofoam.ui.form_schema).
    regionSolvers: dict[str, str] = Field(min_length=1, json_schema_extra={"i18n": "nf.region"})


class RelativeConvergenceConfig(BaseModel):
    """Converged once the exchanged data of field ``data`` changes by less than ``limit``."""

    type: Literal["relative"]
    data: str
    limit: float = Field(gt=0)


class AccelerationConfig(BaseModel):
    """How a side relaxes what it receives: by a ``constant`` factor, or ``aitken``."""

    type: Literal["constant", "aitken"]
    relaxation: float = Field(gt=0, le=1)


@IOStrategy(YAML("system/regionCoupling.yaml"))
class RegionCouplingConfig(BaseConfig):
    """``system/regionCoupling.yaml`` — the interface between two regions and how they are coupled.

    ``interface`` names the coupled patch of each of the two regions; their
    solutions repeat in one coupling loop until every ``convergence`` measure
    holds, or ``maxIterations`` passes. ``serial`` lets the second region receive
    what the first sent in the same pass, ``parallel`` what it sent in the
    previous one. Without the file the regions are not coupled.

    Example::

        interface:
          solid: interface
          fluid: interface
        scheme: serial
        mapping: nearest-neighbor
        acceleration:
          type: aitken
          relaxation: 0.5
        convergence:
          - type: relative
            data: T
            limit: 1.0e-5
    """

    interface: dict[str, str] = Field(min_length=2, max_length=2)
    scheme: Literal["serial", "parallel"] = "serial"
    mapping: Literal["nearest-neighbor"] = "nearest-neighbor"
    acceleration: AccelerationConfig = AccelerationConfig(type="constant", relaxation=1.0)
    convergence: list[RelativeConvergenceConfig] = Field(min_length=1)
    maxIterations: int = Field(default=100, gt=0)
