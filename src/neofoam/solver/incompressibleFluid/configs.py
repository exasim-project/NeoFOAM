# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-core configs for incompressibleFluid.

Pydantic ``BaseConfig`` for ``system/controlDict``, loaded via
``@IOStrategy(OF(...))`` — it feeds validation of the time-stepping controls
before the solver opens any C++ runtime. ``constant/transportProperties`` is
**not** declared here: it is owned by the viscosity model
(:class:`neofoam.viscosity.config.TransportPropertiesConfig`), which the solver
binds as a core model family.
"""

from typing import Optional

from pydantic import Field, model_validator

from neofoam.io import BaseConfig, IOStrategy, OF


@IOStrategy(OF("system/controlDict"))
class ControlDictConfig(BaseConfig):
    """Time-stepping and output control from ``system/controlDict``."""

    application: str = "pimpleFoam"
    endTime: float = Field(gt=0)
    deltaT: float = Field(gt=0)
    adjustTimeStep: bool = False
    maxCo: Optional[float] = Field(default=None, gt=0)
    writeControl: str = "timeStep"
    writeInterval: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def check_adjustTimeStep(self) -> "ControlDictConfig":
        if self.adjustTimeStep and self.maxCo is None:
            raise ValueError("maxCo must be set when adjustTimeStep=True")
        return self
