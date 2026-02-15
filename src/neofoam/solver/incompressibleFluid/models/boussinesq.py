# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Boussinesq model plugin for incompressibleFluid solver."""

from typing import Annotated, Any

import pybFoam as pyf
from pybFoam import fvm, surfaceScalarField, volScalarField

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import field

from .incompressibleFluidModel import Model, incompressibleFluidModel


boussinesq = Model("boussinesq").register_with(incompressibleFluidModel)
boussinesq.beta = 3e-3
boussinesq.TRef = 300.0
boussinesq.Pr = 0.7
boussinesq.Prt = 0.85
boussinesq.hRef = 0.0


@boussinesq.detect
def detect_model() -> bool:
    try:
        props = pyf.dictionary.read("constant/transportProperties")
        toc = list(props.toc())
        return "beta" in toc and "TRef" in toc
    except Exception:
        return False


@boussinesq.load
def load_config() -> None:
    props = pyf.dictionary.read("constant/transportProperties")
    boussinesq.beta = props.get_scalar("beta")
    boussinesq.TRef = props.get_scalar("TRef")
    boussinesq.Pr = props.get_scalar("Pr")
    boussinesq.Prt = props.get_scalar("Prt")


@boussinesq.resolve
def resolve(config: Any) -> None:
    pressure_model = config.get("pressureVelocity")
    if pressure_model is not None:
        pressure_model.use_boussinesq = True


@boussinesq.build
def build() -> list[Any]:
    def create_T(context: dict[str, Any]) -> Any:
        return volScalarField.read_field(context["mesh"], "T")

    def create_alphat(context: dict[str, Any]) -> Any:
        return volScalarField.read_field(context["mesh"], "alphat")

    def create_rhok(context: dict[str, Any]) -> Any:
        return volScalarField.read_field(context["mesh"], "rhok")

    def create_gh(context: dict[str, Any]) -> Any:
        mesh = context["mesh"]
        g = pyf.uniformDimensionedVectorField(mesh, "g")
        g_value = g.value()
        g_mag = (g_value[0] ** 2 + g_value[1] ** 2 + g_value[2] ** 2) ** 0.5
        ghRef = g_mag * boussinesq.hRef if g_mag > 1e-15 else 0.0
        gh_ref_dim = pyf.dimensionedScalar(
            "ghRef", g.dimensions() * pyf.dimLength, ghRef
        )
        return volScalarField(pyf.Word("gh"), (g & mesh.C()) - gh_ref_dim)

    def create_ghf(context: dict[str, Any]) -> Any:
        mesh = context["mesh"]
        g = pyf.uniformDimensionedVectorField(mesh, "g")
        g_value = g.value()
        g_mag = (g_value[0] ** 2 + g_value[1] ** 2 + g_value[2] ** 2) ** 0.5
        ghRef = g_mag * boussinesq.hRef if g_mag > 1e-15 else 0.0
        gh_ref_dim = pyf.dimensionedScalar(
            "ghRef", g.dimensions() * pyf.dimLength, ghRef
        )
        return surfaceScalarField(pyf.Word("ghf"), (g & mesh.Cf()) - gh_ref_dim)

    def create_p_rgh(context: dict[str, Any]) -> Any:
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
    T: Any,
    phi: Any,
    turbulence: Annotated[Any, "models"],
    alphat: Any,
) -> FieldUpdates:
    pr = pyf.dimensionedScalar("Pr", pyf.dimless, boussinesq.Pr)
    prt = pyf.dimensionedScalar("Prt", pyf.dimless, boussinesq.Prt)

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
def update_rhok(T: Any, rhok: Any) -> FieldUpdates:
    beta = pyf.dimensionedScalar(
        "beta", pyf.dimless / pyf.dimTemperature, boussinesq.beta
    )
    t_ref = pyf.dimensionedScalar("TRef", pyf.dimTemperature, boussinesq.TRef)
    one = pyf.dimensionedScalar("one", pyf.dimless, 1.0)

    rhok.assign(one - beta * (T - t_ref))
    return FieldUpdates({"rhok": rhok})
