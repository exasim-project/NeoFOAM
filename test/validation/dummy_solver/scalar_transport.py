# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Scalar transport model — optional, discovered via plugin system.

Challenge: adds a new transported scalar field (T) with its own
ddt/div/laplacian entries and solver requirement. Has Field(gt=0)
constraints on physical parameters (Pr, Prt are divisors).
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


class ScalarTransportConfig(BaseConfig):
    beta: float = 3e-3
    TRef: float = 300.0
    Pr: float = Field(default=0.7, gt=0)
    Prt: float = Field(default=0.85, gt=0)


scalar_transport = Model("scalar_transport").register_with(DummySolverModel)


@scalar_transport.detect
def detect(_case_dir: Path) -> bool:
    return True


@scalar_transport.load
def load(_case_dir: Path, _entry: Any) -> ScalarTransportConfig:
    return ScalarTransportConfig()


@scalar_transport.resolve
def resolve(self: Any, ctx: ConfigContext) -> None:
    core = ctx.get("core_algorithm")
    if core is not None:
        core.has_scalar_transport = True


@scalar_transport.operation(operation_number="2.5", depends_on=["momentum"])
@fvSchemes.add(ddt="ddt(T)", div="div(phi,T)", laplacian="laplacian(alphaEff,T)")
@fvSolution.add("T")
def energy_equation(self: Any) -> FieldUpdates:
    return FieldUpdates({})
