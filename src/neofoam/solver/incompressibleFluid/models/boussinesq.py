# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Boussinesq optional model for the incompressibleFluid solver.

Adds buoyancy-driven flow on top of PIMPLE: temperature T, thermal
turbulent diffusivity alphat, density factor rhok, geopotential gh/ghf,
modified pressure p_rgh, plus the energy equation and rhok update ops.

Ported from ``feat/python_solvers`` and adapted to the ModelSpec API:
build receives the loaded config (no ``self``), and the
``fvSchemes`` / ``fvSolution`` decorators from ``neofoam.foam`` (not yet
present in ``stack/python_arch``) are omitted.
"""

from pathlib import Path
from typing import Annotated, Any, Protocol

import pybFoam as pyf
from pybFoam import fvm, surfaceScalarField, volScalarField
from pydantic import Field

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import ConfigContext, field
from neofoam.io import BaseConfig

from .incompressibleFluidModel import Model, incompressibleFluidModel


class ThermalTurbulenceModel(Protocol):
    def nut(self) -> Any: ...

    def nu(self) -> Any: ...


class BoussinesqConfig(BaseConfig):
    beta: float
    TRef: float
    Pr: float = Field(default=0.7, gt=0)
    Prt: float = Field(default=0.85, gt=0)
    hRef: float = 0.0


def _read_boussinesq_config() -> BoussinesqConfig:
    """Read transportProperties and construct a validated BoussinesqConfig.

    ``beta`` and ``TRef`` are required and read directly; ``Pr``, ``Prt``,
    ``hRef`` keep their schema defaults if not present in the dict.
    """
    props = pyf.dictionary.read("constant/transportProperties")
    values: dict[str, float] = {
        "beta": props.get[float]("beta"),
        "TRef": props.get[float]("TRef"),
    }
    if props.found("Pr"):
        values["Pr"] = props.get[float]("Pr")
    if props.found("Prt"):
        values["Prt"] = props.get[float]("Prt")
    if props.found("hRef"):
        values["hRef"] = props.get[float]("hRef")
    return BoussinesqConfig(**values)


boussinesq = Model("boussinesq").register_with(incompressibleFluidModel)


@boussinesq.load
def load(_case_dir: Path, _instance_id: str) -> BoussinesqConfig:
    return _read_boussinesq_config()


@boussinesq.detect
def detect_model() -> bool:
    """Boussinesq is active when transportProperties has both beta and TRef."""
    try:
        props = pyf.dictionary.read("constant/transportProperties")
        return bool(props.found("beta") and props.found("TRef"))
    except Exception:
        return False


@boussinesq.resolve
def resolve(_config: BoussinesqConfig, ctx: ConfigContext) -> None:
    """Flip ``use_boussinesq`` on the pressure-velocity algorithm spec.

    The algorithm spec is registered in ConfigContext under its ``name``
    (e.g. ``"Pimple"``). Setting this flag steers
    :func:`pimpleAlgorithm.collected_operations` to dispatch the
    boussinesq variants and adds ``p_rgh`` to the pressure-reference
    dependency list.
    """
    for algo_name in ("Pimple", "Simple", "Piso"):
        pressure_model = ctx.get(algo_name) if ctx.contains(algo_name) else None
        if pressure_model is not None:
            pressure_model.use_boussinesq = True
            return


@boussinesq.build
def build(configs: BoussinesqConfig) -> list[object]:
    def create_T(context: dict[str, Any]) -> volScalarField:
        return volScalarField.read_field(context["mesh"], "T")

    def create_alphat(context: dict[str, Any]) -> volScalarField:
        return volScalarField.read_field(context["mesh"], "alphat")

    def create_rhok(context: dict[str, Any]) -> volScalarField:
        return volScalarField.read_field(context["mesh"], "rhok")

    def _gh_ref(context: dict[str, Any]) -> tuple[Any, Any]:
        mesh = context["mesh"]
        g = pyf.uniformDimensionedVectorField(mesh, "g")
        g_value = g.value()
        g_mag = (g_value[0] ** 2 + g_value[1] ** 2 + g_value[2] ** 2) ** 0.5
        gh_ref = g_mag * configs.hRef if g_mag > 1e-15 else 0.0
        gh_ref_dim = pyf.dimensionedScalar(
            "ghRef", g.dimensions() * pyf.dimLength, gh_ref
        )
        return g, gh_ref_dim

    def create_gh(context: dict[str, Any]) -> volScalarField:
        mesh = context["mesh"]
        g, gh_ref_dim = _gh_ref(context)
        return volScalarField(pyf.Word("gh"), (g & mesh.C()) - gh_ref_dim)

    def create_ghf(context: dict[str, Any]) -> surfaceScalarField:
        mesh = context["mesh"]
        g, gh_ref_dim = _gh_ref(context)
        return surfaceScalarField(pyf.Word("ghf"), (g & mesh.Cf()) - gh_ref_dim)

    def create_p_rgh(context: dict[str, Any]) -> volScalarField:
        mesh = context["mesh"]
        mesh.setFluxRequired(pyf.Word("p_rgh"))
        return volScalarField.read_field(mesh, "p_rgh")

    return [
        field("T", create_T, depends_on=["mesh"]),
        field("alphat", create_alphat, depends_on=["mesh"]),
        field("rhok", create_rhok, depends_on=["mesh", "fields.T"]),
        field("gh", create_gh, depends_on=["mesh"]),
        field("ghf", create_ghf, depends_on=["mesh"]),
        field(
            "p_rgh",
            create_p_rgh,
            depends_on=["mesh", "fields.p", "fields.rhok", "fields.gh"],
        ),
    ]


@boussinesq.operation(operation_number="2.5", depends_on=["momentum"])
def solve_energy(
    self: Any,
    T: volScalarField,
    phi: surfaceScalarField,
    turbulence: Annotated[ThermalTurbulenceModel, "models"],
    alphat: volScalarField,
) -> FieldUpdates:
    configs: BoussinesqConfig = self.config
    pr = pyf.dimensionedScalar("Pr", pyf.dimless, configs.Pr)
    prt = pyf.dimensionedScalar("Prt", pyf.dimless, configs.Prt)

    alphat.assign(turbulence.nut() / prt)
    alphat.correctBoundaryConditions()

    alpha_eff = turbulence.nu() / pr + alphat
    t_eqn = pyf.fvScalarMatrix(
        fvm.ddt(T) + fvm.div(phi, T) - fvm.laplacian(alpha_eff, T)
    )
    t_eqn.relax()
    t_eqn.solve()

    return FieldUpdates({"T": T, "alphat": alphat})


@boussinesq.operation(operation_number="2.7", depends_on=["solve_energy"])
def update_rhok(self: Any, T: volScalarField, rhok: volScalarField) -> FieldUpdates:
    configs: BoussinesqConfig = self.config
    beta = pyf.dimensionedScalar("beta", pyf.dimless / pyf.dimTemperature, configs.beta)
    t_ref = pyf.dimensionedScalar("TRef", pyf.dimTemperature, configs.TRef)
    one = pyf.dimensionedScalar("one", pyf.dimless, 1.0)

    rhok.assign(one - beta * (T - t_ref))
    return FieldUpdates({"rhok": rhok})
