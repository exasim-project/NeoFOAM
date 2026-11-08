# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case config of the multiRun solver: the shared time loop, the region solvers, their coupling."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.framework.coupling import ACCELERATIONS, CONVERGENCES, MAPPINGS
from neofoam.io import YAML, BaseConfig, IOStrategy


def _known(name: str, registry: dict[str, Any], what: str) -> str:
    if name not in registry:
        raise ValueError(f"unknown {what} '{name}'; known: {sorted(registry)}")
    return name


@IOStrategy(YAML("system/controlDict.yaml"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict.yaml`` — the time control every region reads, and its solvers.

    ``regionSolvers`` maps each region to the solver that runs it, in stepping
    order (a coupling group's last region leads it); each region keeps its own
    files under ``constant/<region>/``, ``0/<region>/`` and ``system/<region>/``.
    Every region runs its own time loop on this file's times.

    Example::

        endTime: 10.0
        deltaT: 1.0
        regionSolvers:
          fluid: myFluidSolver
          solid: mySolidSolver
    """

    # The i18n prefix labels the wizard's "add a region" row (neofoam.ui.form_schema).
    regionSolvers: dict[str, str] = Field(min_length=1, json_schema_extra={"i18n": "nf.region"})


class ConvergenceConfig(BaseModel):
    """A convergence measure on the exchanged data of field ``data`` with its ``limit``.

    ``type`` names it in :data:`~neofoam.framework.coupling.CONVERGENCES`
    (``relative``: the data changes by less than ``limit`` relative to it).
    """

    type: str
    data: str
    limit: float = Field(gt=0)

    @field_validator("type")
    @classmethod
    def _registered(cls, value: str) -> str:
        return _known(value, CONVERGENCES, "convergence")


class AccelerationConfig(BaseModel):
    """How the leader relaxes the data carried into a pass, with its relaxation factor.

    ``type`` names it in :data:`~neofoam.framework.coupling.ACCELERATIONS`
    (``constant`` factor, or ``aitken`` starting from it).
    """

    type: str
    relaxation: float = Field(gt=0, le=1)

    @field_validator("type")
    @classmethod
    def _registered(cls, value: str) -> str:
        return _known(value, ACCELERATIONS, "acceleration")


class ExchangeConfig(BaseModel):
    """One exchange: the coupled patch of each of its two regions, and how data is mapped.

    ``sides`` maps each region to its patch; ``name`` defaults to the regions
    joined by ``-``. ``mapping`` maps the point data of one side onto the other's
    points, a kind of :data:`~neofoam.framework.coupling.MAPPINGS`:
    ``nearest-neighbor`` (both patches need coordinates) or ``identity`` (both
    patches share their points).
    """

    name: Optional[str] = None
    sides: dict[str, str] = Field(min_length=2, max_length=2)
    mapping: str = "nearest-neighbor"

    @field_validator("mapping")
    @classmethod
    def _registered(cls, value: str) -> str:
        return _known(value, MAPPINGS, "mapping")


@IOStrategy(YAML("system/regionCoupling.yaml"))
class RegionCouplingConfig(BaseConfig):
    """``system/regionCoupling.yaml`` — the exchanges between regions and how they are coupled.

    Each entry of ``exchanges`` joins the coupled patches of two regions. Exchanges
    that share a region form one group; every region of a group repeats its
    solution in its coupling loop until every ``convergence`` measure holds, or
    ``maxIterations`` passes, as the group's leader decides. ``serial`` lets a
    region receive what a region before it sent in the same pass, ``parallel``
    what it sent in the previous one. Without the file the
    regions are not coupled.

    Example::

        exchanges:
          - sides: {fluid: wall, solid: right}
            mapping: nearest-neighbor
        scheme: serial
        acceleration:
          type: aitken
          relaxation: 0.5
        convergence:
          - type: relative
            data: T
            limit: 1.0e-5
    """

    exchanges: list[ExchangeConfig] = Field(min_length=1)
    scheme: Literal["serial", "parallel"] = "serial"
    acceleration: AccelerationConfig = AccelerationConfig(type="constant", relaxation=1.0)
    convergence: list[ConvergenceConfig] = Field(min_length=1)
    maxIterations: int = Field(default=100, gt=0)
