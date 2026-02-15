# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Boussinesq model plugin for incompressibleFluid solver."""

from typing import Annotated, Optional, Protocol

import pybFoam as pyf
from pybFoam import fvm, surfaceScalarField, volScalarField

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import field
from neofoam.io import BaseConfig

from .incompressibleFluidModel import Model, incompressibleFluidModel


class ThermalTurbulenceModel(Protocol):
    def nut(self) -> object: ...

    def nu(self) -> object: ...


class BoussinesqConfig(BaseConfig):
    beta: float = 3e-3
    TRef: float = 300.0
    Pr: float = 0.7
    Prt: float = 0.85
    hRef: float = 0.0


def load_boussinesq_config(_case_dir: Optional[object] = None) -> BoussinesqConfig:
    config = BoussinesqConfig()
    props = pyf.dictionary.read("constant/transportProperties")
    toc = set(props.toc())

    if "beta" in toc:
        config.beta = props.get_scalar("beta")
    if "TRef" in toc:
        config.TRef = props.get_scalar("TRef")
    if "Pr" in toc:
        config.Pr = props.get_scalar("Pr")
    if "Prt" in toc:
        config.Prt = props.get_scalar("Prt")

    return config


boussinesq = (
    Model("boussinesq")
    .with_config(BoussinesqConfig, name="configs", loader=load_boussinesq_config)
    .register_with(incompressibleFluidModel)
)


@boussinesq.detect
def detect_model() -> bool:
    try:
        props = pyf.dictionary.read("constant/transportProperties")
        toc = set(props.toc())
        return "beta" in toc and "TRef" in toc
    except Exception:
        return False


@boussinesq.resolve
def resolve(config: BoussinesqConfig) -> None:
    """Set use_boussinesq flag on the pressure-velocity algorithm model."""
    # Access the pressure model from the init's core_models
    # The pressure model is always the first core model
    from ..create_fields import init

    if init.core_models and len(init.core_models) > 0:
        pressure_model = init.core_models[0]
        pressure_model.use_boussinesq = True


@boussinesq.build
def build() -> list[object]:
    configs = boussinesq.config("configs")

    def create_T(context: dict[str, object]) -> volScalarField:
        return volScalarField.read_field(context["mesh"], "T")

    def create_alphat(context: dict[str, object]) -> volScalarField:
        return volScalarField.read_field(context["mesh"], "alphat")

    def create_rhok(context: dict[str, object]) -> volScalarField:
        return volScalarField.read_field(context["mesh"], "rhok")

    def create_gh(context: dict[str, object]) -> volScalarField:
        mesh = context["mesh"]
        g = pyf.uniformDimensionedVectorField(mesh, "g")
        g_value = g.value()
        g_mag = (g_value[0] ** 2 + g_value[1] ** 2 + g_value[2] ** 2) ** 0.5
        gh_ref = g_mag * configs.hRef if g_mag > 1e-15 else 0.0
        gh_ref_dim = pyf.dimensionedScalar(
            "ghRef", g.dimensions() * pyf.dimLength, gh_ref
        )
        return volScalarField(pyf.Word("gh"), (g & mesh.C()) - gh_ref_dim)

    def create_ghf(context: dict[str, object]) -> surfaceScalarField:
        mesh = context["mesh"]
        g = pyf.uniformDimensionedVectorField(mesh, "g")
        g_value = g.value()
        g_mag = (g_value[0] ** 2 + g_value[1] ** 2 + g_value[2] ** 2) ** 0.5
        gh_ref = g_mag * configs.hRef if g_mag > 1e-15 else 0.0
        gh_ref_dim = pyf.dimensionedScalar(
            "ghRef", g.dimensions() * pyf.dimLength, gh_ref
        )
        return surfaceScalarField(pyf.Word("ghf"), (g & mesh.Cf()) - gh_ref_dim)

    def create_p_rgh(context: dict[str, object]) -> volScalarField:
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
    T: volScalarField,
    phi: surfaceScalarField,
    turbulence: Annotated[ThermalTurbulenceModel, "models"],
    alphat: volScalarField,
) -> FieldUpdates:
    configs = boussinesq.config("configs")
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
def update_rhok(T: volScalarField, rhok: volScalarField) -> FieldUpdates:
    configs = boussinesq.config("configs")
    beta = pyf.dimensionedScalar("beta", pyf.dimless / pyf.dimTemperature, configs.beta)
    t_ref = pyf.dimensionedScalar("TRef", pyf.dimTemperature, configs.TRef)
    one = pyf.dimensionedScalar("one", pyf.dimless, 1.0)

    rhok.assign(one - beta * (T - t_ref))
    return FieldUpdates({"rhok": rhok})
