# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Core algorithm model — always active, NOT in plugin system.

Challenge: covers all 6 standard fvSchemes sections (ddt, div, grad,
laplacian, snGrad, interpolation) with concrete entry keys.
Registered via init.register_core_models().
"""

from pathlib import Path
from typing import Any

from pydantic import Field

from neofoam.foam import fvSchemes, fvSolution
from neofoam.framework.context import FieldUpdates
from neofoam.framework.model import Model
from neofoam.io import BaseConfig


class CoreAlgorithmConfig(BaseConfig):
    nCorrectors: int = Field(default=2, ge=1)
    nOuterCorrectors: int = Field(default=1, ge=1)


core_algorithm = Model("core_algorithm")


@core_algorithm.load
def load(_case_dir: Path, _entry: Any) -> CoreAlgorithmConfig:
    return CoreAlgorithmConfig()


@core_algorithm.operation(operation_number="2.1")
@fvSchemes.add(ddt="ddt(U)", div="div(phi,U)", grad="grad(U)", laplacian="laplacian(nuEff,U)")
@fvSolution.add("U")
def momentum(self: Any) -> FieldUpdates:
    return FieldUpdates({})


@core_algorithm.operation(operation_number="2.2", depends_on=["momentum"])
@fvSchemes.add(
    grad="grad(p)",
    laplacian="laplacian(rAU,p)",
    interpolation=["flux(HbyA)", "interpolate(rAU)"],
    snGrad="snGrad(p)",
)
@fvSolution.add("p")
def pressure_correction(self: Any) -> FieldUpdates:
    return FieldUpdates({})
