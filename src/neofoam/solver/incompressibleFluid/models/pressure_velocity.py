# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pressure-velocity model plugin for incompressibleFluid solver."""

from typing import Annotated, Any

import pybFoam as pyf
from pybFoam import volScalarField, volVectorField

from neofoam.algorithms.pressure_velocity import (
    momentum_boussinesq_helper,
    momentum_helper,
    continuity_boussinesq_helper,
    continuity_helper,
)
from neofoam.foam.initialization import read_vol_field
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import field, model

from .incompressibleFluidModel import Model, incompressibleFluidModel


pressure_velocity = Model("pressureVelocity").register_with(incompressibleFluidModel)
pressure_velocity.algorithm_type = "PIMPLE"
pressure_velocity.fv_solution = None
pressure_velocity.pRefCell = None
pressure_velocity.pRefValue = None
pressure_velocity.use_boussinesq = False


@pressure_velocity.detect
def detect_model() -> bool:
    return True


@pressure_velocity.load
def load_config() -> None:
    pressure_velocity.use_boussinesq = False
    pressure_velocity.pRefCell = None
    pressure_velocity.pRefValue = None

    fv_solution = pyf.dictionary.read("system/fvSolution")
    toc = fv_solution.toc()
    if "PIMPLE" in toc:
        pressure_velocity.algorithm_type = "PIMPLE"
    elif "PISO" in toc:
        pressure_velocity.algorithm_type = "PISO"
    elif "SIMPLE" in toc:
        pressure_velocity.algorithm_type = "SIMPLE"
    else:
        raise ValueError("No supported algorithm found in system/fvSolution")
    pressure_velocity.fv_solution = fv_solution


@pressure_velocity.resolve
def resolve(_config: Any) -> None:
    pass


@pressure_velocity.build
def build() -> list[Any]:
    def create_phi(context: dict[str, Any]) -> Any:
        return pyf.createPhi(context["fields.U"])

    def create_pimple_control(context: dict[str, Any]) -> Any:
        return pyf.pimpleControl(context["mesh"])

    def create_cumulative_cont_err(_context: dict[str, Any]) -> list[float]:
        return [0.0]

    return [
        read_vol_field(volScalarField, "p"),
        read_vol_field(volVectorField, "U"),
        field("phi", create_phi, depends_on=["fields.U"]),
        model("pimple_control", create_pimple_control, depends_on=["mesh"]),
        model("cumulativeContErr", create_cumulative_cont_err),
    ]


def ensure_pressure_reference(p: Any, mesh: Any, p_rgh: Any = None) -> None:
    if (
        pressure_velocity.pRefCell is not None
        and pressure_velocity.pRefValue is not None
    ):
        return

    if pressure_velocity.fv_solution is None:
        pressure_velocity.fv_solution = pyf.dictionary.read("system/fvSolution")

    algo_dict = pressure_velocity.fv_solution.subDict(pressure_velocity.algorithm_type)
    pressure_field = p_rgh if p_rgh is not None else p
    field_name = "p_rgh" if p_rgh is not None else "p"

    if not (
        algo_dict.found(f"{field_name}RefCell")
        or algo_dict.found(f"{field_name}RefPoint")
    ):
        if p_rgh is not None and (
            algo_dict.found("pRefCell") or algo_dict.found("pRefPoint")
        ):
            pRefCell, pRefValue = pyf.setRefCell(p, algo_dict, True)
        else:
            pRefCell, pRefValue = pyf.setRefCell(pressure_field, algo_dict)
    else:
        pRefCell, pRefValue = pyf.setRefCell(pressure_field, algo_dict)

    pressure_velocity.pRefCell = pRefCell
    pressure_velocity.pRefValue = pRefValue

    mesh.setFluxRequired(pyf.Word("p"))
    if p_rgh is not None:
        mesh.setFluxRequired(pyf.Word("p_rgh"))


def inner_loop(ctx: Any) -> bool:
    return bool(ctx.models["pimple_control"].loop())


pressure_velocity.inner_loop = inner_loop


@pressure_velocity.operation(operation_number="2.1")
def momentum(
    U: Any,
    phi: Any,
    p: Any,
    turbulence: Annotated[Any, "models"],
    pimple_control: Annotated[Any, "models"],
    p_rgh: Any = None,
    rhok: Any = None,
    ghf: Any = None,
) -> FieldUpdates:
    if pressure_velocity.use_boussinesq:
        if p_rgh is None or rhok is None or ghf is None:
            raise RuntimeError("Boussinesq mode requires p_rgh, rhok, and ghf fields")
        ensure_pressure_reference(p, U.mesh(), p_rgh)
        UEqn = momentum_boussinesq_helper(
            U=U,
            phi=phi,
            p_rgh=p_rgh,
            rhok=rhok,
            ghf=ghf,
            turbulence=turbulence,
            pimple_control=pimple_control,
        )
    else:
        ensure_pressure_reference(p, U.mesh())
        UEqn = momentum_helper(
            U=U,
            phi=phi,
            p=p,
            turbulence=turbulence,
            pimple_control=pimple_control,
        )

    return FieldUpdates({"UEqn": UEqn, "U": U})


@pressure_velocity.operation(operation_number="2.2", depends_on=["momentum"])
def continuity(
    U: Any,
    p: Any,
    phi: Any,
    UEqn: Any,
    pimple_control: Annotated[Any, "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    p_rgh: Any = None,
    rhok: Any = None,
    gh: Any = None,
    ghf: Any = None,
) -> FieldUpdates:
    if pressure_velocity.use_boussinesq:
        if p_rgh is None or rhok is None or gh is None or ghf is None:
            raise RuntimeError(
                "Boussinesq mode requires p_rgh, rhok, gh, and ghf fields"
            )
        ensure_pressure_reference(p, U.mesh(), p_rgh)
        continuity_boussinesq_helper(
            U=U,
            p=p,
            p_rgh=p_rgh,
            phi=phi,
            UEqn=UEqn,
            rhok=rhok,
            gh=gh,
            ghf=ghf,
            pimple_control=pimple_control,
            cumulativeContErr=cumulativeContErr,
            pRefCell=pressure_velocity.pRefCell,
            pRefValue=pressure_velocity.pRefValue,
        )
        return FieldUpdates({"U": U, "p": p, "phi": phi, "p_rgh": p_rgh})

    ensure_pressure_reference(p, U.mesh())
    continuity_helper(
        U=U,
        p=p,
        phi=phi,
        UEqn=UEqn,
        pimple_control=pimple_control,
        cumulativeContErr=cumulativeContErr,
        pRefCell=pressure_velocity.pRefCell,
        pRefValue=pressure_velocity.pRefValue,
    )
    return FieldUpdates({"U": U, "p": p, "phi": phi})
