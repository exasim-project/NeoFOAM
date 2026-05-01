# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Wall model — optional, discovered via plugin system.

Challenge: adds a non-standard fvSchemes section (wallDist) plus extra
div/grad/laplacian entries for its own field. Has Field(gt=0) constraints
on divisor coefficients.
"""

from pathlib import Path
from typing import Any

from pydantic import Field

from neofoam.foam import fvSchemes, fvSolution
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import ConfigContext
from neofoam.framework.model import Model
from neofoam.io import BaseConfig

from .plugin import DummySolverModel


class WallModelConfig(BaseConfig):
    sigma: float = Field(default=2.0 / 3.0, gt=0)
    kappa: float = Field(default=0.41, gt=0)


wall_model = Model("wall_model").register_with(DummySolverModel)


@wall_model.detect
def detect(_case_dir: Path) -> bool:
    return True


@wall_model.load
def load(_case_dir: Path, _entry: Any) -> WallModelConfig:
    return WallModelConfig()


@wall_model.resolve
def resolve(self: Any, ctx: ConfigContext) -> None:
    core = ctx.get("core_algorithm")
    if core is not None:
        core.has_wall_model = True


@wall_model.operation(operation_number="5", depends_on=["pressure_correction"])
@fvSchemes.add(
    ddt="ddt(nuTilda)",
    div="div(phi,nuTilda)",
    grad="grad(nuTilda)",
    laplacian="laplacian(DnuTildaEff,nuTilda)",
    wallDist="method",
)
@fvSolution.add("nuTilda")
def wall_solve(self: Any) -> FieldUpdates:
    return FieldUpdates({})
