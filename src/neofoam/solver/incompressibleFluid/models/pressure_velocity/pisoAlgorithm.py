# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""PISO pressure-velocity algorithm model (not implemented)."""

from typing import Any

from pydantic import Field

from neofoam.framework.context import FieldUpdates
from neofoam.io import BaseConfig, IOStrategy, OF

from ..incompressibleFluidModel import Model


@IOStrategy(OF("system/fvSolution", subdict="PISO"))
class PisoConfig(BaseConfig):
    """PISO pressure-velocity coupling controls."""

    nCorrectors: int = Field(default=2, ge=1)
    nNonOrthogonalCorrectors: int = Field(default=0, ge=0)
    momentumPredictor: bool = True


piso = Model("Piso")


def build() -> list[Any]:
    raise NotImplementedError("PISO pressure-velocity model is not implemented")


def inner_loop(_ctx: Any) -> bool:
    raise NotImplementedError("PISO pressure-velocity model is not implemented")


def momentum(**_kwargs: Any) -> FieldUpdates:
    raise NotImplementedError("PISO pressure-velocity model is not implemented")


def continuity(**_kwargs: Any) -> FieldUpdates:
    raise NotImplementedError("PISO pressure-velocity model is not implemented")
