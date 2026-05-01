# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-core configs for incompressibleFluid."""

from typing import Optional

from pydantic import Field, model_validator

from neofoam.io import BaseConfig, IOStrategy, OF


@IOStrategy(OF("system/controlDict"))
class ControlDictConfig(BaseConfig):
    """Time stepping and output control from system/controlDict."""

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


@IOStrategy(OF("constant/transportProperties"))
class TransportPropertiesConfig(BaseConfig):
    """Fluid transport properties from constant/transportProperties."""

    transportModel: str = "Newtonian"
    nu: float = Field(gt=0)
