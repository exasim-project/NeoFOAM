# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-core configs (Plan 03)."""

from pydantic import Field

from neofoam.io import BaseConfig


class TimeConfig(BaseConfig):
    """Time stepping parameters."""

    endTime: float = Field(gt=0)
    deltaT: float = Field(gt=0)


class FluidPropertiesConfig(BaseConfig):
    """Fluid material properties."""

    nu: float = Field(gt=0)
