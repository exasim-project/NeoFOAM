# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Adaptive time-stepping (Courant/CFL) as an optional time-step model.

The core solution loop is a fixed-step advancer; stability rules are opt-in.
This model is the first such rule: when selected it caps ``deltaT`` by the flow
Courant number (and, optionally, an absolute ``maxDeltaT``). It owns the
``adjustTimeStep``/``maxCo``/``maxDeltaT`` keys of ``system/controlDict`` (which
left :class:`~neofoam.algorithms.solution_loop.config.TimeControlConfig`), and its
``@build`` installs the constraints into the built loop engine and registers the
``measurement_provider.courant`` the loop body consults each step.

Adding another time-step rule (a VoF interface Courant, a diffusion-number cap,
…) is a sibling module: a ``DeltaTConstraint`` plus a toggle model whose
``@build`` yields it via ``install_constraints_step`` and registers any
``measurement_provider.<name>`` it needs — the core loop is never edited.
"""

from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf
from pybFoam import computeCFLNumber
from pydantic import Field

from neofoam.algorithms.constraints.time_step import (
    CourantConstraint,
    MaxDeltaTConstraint,
)
from neofoam.algorithms.solution_loop.solution_loop import install_constraints_step
from neofoam.framework.initialization import InitStep, model
from neofoam.io import OF, BaseConfig, IOStrategy

from .incompressibleFluidModel import Model, incompressibleFluidModel


@IOStrategy(OF("system/controlDict"))
class CourantControlConfig(BaseConfig):
    """The adaptive-stepping slice of ``system/controlDict``.

    Co-owns the file with
    :class:`~neofoam.solver.incompressibleFluid.configs.ControlDictConfig`; the
    writer merges co-owners into one ``controlDict``. Present only when the
    adaptive time-step model is selected.
    """

    adjustTimeStep: bool = True
    maxCo: float = Field(gt=0)
    maxDeltaT: Optional[float] = Field(default=None, gt=0)


adaptiveTimeStep = (
    Model("adaptiveTimeStep")
    .register_with(incompressibleFluidModel)
    .as_toggle("Adaptive time step (Courant)")
)
adaptiveTimeStep.config(CourantControlConfig)


@adaptiveTimeStep.load
def load(_case_dir: Path, _instance_id: str) -> CourantControlConfig:
    return CourantControlConfig.load(case_dir=_case_dir)


@adaptiveTimeStep.detect
def detect_model() -> bool:
    """Active when ``controlDict`` opts into adaptive stepping."""
    try:
        cd = pyf.dictionary.read("system/controlDict")
        return bool(cd.found("adjustTimeStep") and cd.get[bool]("adjustTimeStep"))
    except Exception:
        return False


@adaptiveTimeStep.build
def build(config: CourantControlConfig) -> list[InitStep]:
    """Install the CFL (+ optional maxDeltaT) constraints and their measurement."""

    def constraints(_ctx: dict[str, Any]) -> Any:
        yield CourantConstraint(maxCo=float(config.maxCo))
        if config.maxDeltaT is not None:
            yield MaxDeltaTConstraint(maxDeltaT=float(config.maxDeltaT))

    def courant_provider(ctx: Any) -> float:
        """The flow Courant number on the live ``phi`` (pybFoam)."""
        return float(computeCFLNumber(ctx.fields["phi"])[0])

    return [
        install_constraints_step("adaptive_time_step", constraints),
        model("measurement_provider.courant", lambda _ctx: courant_provider),
    ]
