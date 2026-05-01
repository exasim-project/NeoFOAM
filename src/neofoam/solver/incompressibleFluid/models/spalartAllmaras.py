# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spalart-Allmaras turbulence model plugin for incompressibleFluid solver."""

from pathlib import Path
from typing import Annotated, Any

import pybFoam as pyf

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import ConfigContext, field
from pydantic import Field

from neofoam.foam import fvSchemes, fvSolution
from neofoam.io import BaseConfig

from .incompressibleFluidModel import Model, incompressibleFluidModel


class SpalartAllmarasConfig(BaseConfig):
    """SA model constants (standard SA-neg values from Spalart & Allmaras 1992)."""

    Cb1: float = 0.1355
    Cb2: float = 0.622
    Cw2: float = 0.3
    Cw3: float = 2.0
    Cv1: float = Field(default=7.1, ge=0)
    sigma: float = Field(default=2.0 / 3.0, gt=0)
    kappa: float = Field(default=0.41, gt=0)
    Cs: float = 0.3

    @property
    def Cw1(self) -> float:
        return self.Cb1 / self.kappa**2 + (1 + self.Cb2) / self.sigma


spalart_allmaras = Model("spalart_allmaras").register_with(incompressibleFluidModel)


@spalart_allmaras.detect
def detect(_case_dir: Path) -> bool:
    try:
        props = pyf.dictionary.read("constant/turbulenceProperties")
        if not props.found("RAS"):
            return False
        ras = props.subDict("RAS")
        return bool(
            ras.found("RASModel") and ras.get_word("RASModel") == "SpalartAllmaras"
        )
    except Exception:
        return False


@spalart_allmaras.load
def load(_case_dir: Path, _entry: Any) -> SpalartAllmarasConfig:
    return SpalartAllmarasConfig()


@spalart_allmaras.resolve
def resolve(self: Any, ctx: ConfigContext) -> None:
    """Set turbulence_type on the pressure-velocity algorithm model."""
    for algo_name in ("Pimple", "Simple", "Piso"):
        pressure_model = ctx.get(algo_name)
        if pressure_model is not None:
            pressure_model.turbulence_type = "spalart_allmaras"
            return


@spalart_allmaras.build
def build(self: Any, configs: SpalartAllmarasConfig) -> list[object]:
    """Register nuTilda, nut, and wall distance fields."""

    def create_nuTilda(context: dict[str, Any]) -> Any:
        mesh = context["mesh"]
        return pyf.volScalarField.read_field(mesh, "nuTilda")

    def create_nut(context: dict[str, Any]) -> Any:
        mesh = context["mesh"]
        return pyf.volScalarField.read_field(mesh, "nut")

    def create_d(context: dict[str, Any]) -> Any:
        mesh = context["mesh"]
        return pyf.wallDist.New(mesh).y()

    return [
        field("nuTilda", create_nuTilda, depends_on=["mesh"]),
        field("nut", create_nut, depends_on=["mesh"]),
        field("d", create_d, depends_on=["mesh"]),
    ]


@spalart_allmaras.operation(operation_number="5", depends_on=["continuity"])
@fvSchemes.add(
    ddt="ddt(nuTilda)",
    div="div(phi,nuTilda)",
    grad="grad(nuTilda)",
    laplacian="laplacian(DnuTildaEff,nuTilda)",
    wallDist="method",
)
@fvSolution.add("nuTilda")
def turbulence_correction(
    self: Any,
    turbulence: Annotated[Any, "models"],
    nuTilda: Any,
    nut: Any,
) -> FieldUpdates:
    """Correct turbulence fields via OpenFOAM RTS."""
    turbulence.correct()
    return FieldUpdates({"nuTilda": nuTilda, "nut": nut})
