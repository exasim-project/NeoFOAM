# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from typing import Any

import pybFoam as pyf
from pybFoam import (
    fvc,
    fvm,
    fvScalarMatrix,
    fvVectorMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)

from neofoam.foam.initialization import read_vol_field
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import field, model

from ..incompressibleFluidModel import Model
from .base import pressureVelocityAlgorithm

pimple = Model("Pimple").register_with(pressureVelocityAlgorithm)


def ensure_pressure_reference(model_state: Any, p: Any, mesh: Any, p_rgh: Any = None) -> None:
    if model_state.pRefCell is not None and model_state.pRefValue is not None:
        return

    if model_state.fv_solution is None:
        model_state.fv_solution = pyf.dictionary.read("system/fvSolution")

    algo_dict = model_state.fv_solution.subDict(model_state.algorithm_type)
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

    model_state.pRefCell = pRefCell
    model_state.pRefValue = pRefValue

    mesh.setFluxRequired(pyf.Word("p"))
    if p_rgh is not None:
        mesh.setFluxRequired(pyf.Word("p_rgh"))


def momentum_helper(
    U: Any,
    phi: Any,
    p: Any,
    turbulence: Any,
    pimple_control: Any,
) -> Any:
    UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))
    UEqn.relax()

    if pimple_control.momentumPredictor():
        pyf.solve(UEqn + fvc.grad(p))

    return UEqn


def continuity_helper(
    U: Any,
    p: Any,
    phi: Any,
    UEqn: Any,
    pimple_control: Any,
    cumulativeContErr: list[float],
    pRefCell: Any,
    pRefValue: Any,
) -> None:
    while pimple_control.correct():
        rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
        HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p))

        phiHbyA = surfaceScalarField(
            pyf.Word("phiHbyA"),
            fvc.flux(HbyA) + fvc.interpolate(rAU) * fvc.ddtCorr(U, phi),
        )

        pyf.adjustPhi(phiHbyA, U, p)
        pyf.constrainPressure(p, U, phiHbyA, rAU)

        while pimple_control.correctNonOrthogonal():
            pEqn = fvScalarMatrix(fvm.laplacian(rAU, p) - fvc.div(phiHbyA))
            pEqn.setReference(pRefCell, pRefValue, False)
            pEqn.solve(p.select(pimple_control.finalInnerIter()))

            if pimple_control.finalNonOrthogonalIter():
                phi.assign(phiHbyA - pEqn.flux())

        U.assign(HbyA - rAU * fvc.grad(p))
        U.correctBoundaryConditions()

        sum_local, global_err = pyf.computeContinuityErrors(phi)
        cumulativeContErr[0] += global_err
        pyf.Info(
            f"time step continuity errors : sum local = {sum_local}, "
            f"global = {global_err}, cumulative = {cumulativeContErr[0]}"
        )


def momentum_boussinesq_helper(
    U: Any,
    phi: Any,
    p_rgh: Any,
    rhok: Any,
    ghf: Any,
    turbulence: Any,
    pimple_control: Any,
) -> Any:
    mesh = U.mesh()
    UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))
    UEqn.relax()

    if pimple_control.momentumPredictor():
        pyf.solve(
            UEqn
            + fvc.reconstruct(
                (-ghf * fvc.snGrad(rhok) - fvc.snGrad(p_rgh)) * mesh.magSf()
            )
        )

    return UEqn


def continuity_boussinesq_helper(
    U: Any,
    p: Any,
    p_rgh: Any,
    phi: Any,
    UEqn: Any,
    rhok: Any,
    gh: Any,
    ghf: Any,
    pimple_control: Any,
    cumulativeContErr: list[float],
    pRefCell: Any,
    pRefValue: Any,
) -> None:
    mesh = U.mesh()

    while pimple_control.correct():
        rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
        rAUf = surfaceScalarField(pyf.Word("rAUf"), fvc.interpolate(rAU))
        HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p_rgh))

        phig = surfaceScalarField(
            pyf.Word("phig"), -rAUf * ghf * fvc.snGrad(rhok) * mesh.magSf()
        )

        phiHbyA = surfaceScalarField(
            pyf.Word("phiHbyA"),
            fvc.flux(HbyA) + rAUf * fvc.ddtCorr(U, phi) + phig,
        )

        pyf.constrainPressure(p_rgh, U, phiHbyA, rAUf)

        while pimple_control.correctNonOrthogonal():
            pEqn = fvScalarMatrix(fvm.laplacian(rAUf, p_rgh) - fvc.div(phiHbyA))
            pEqn.setReference(pRefCell, pRefValue, False)
            pEqn.solve(p_rgh.select(pimple_control.finalInnerIter()))

            if pimple_control.finalNonOrthogonalIter():
                phi.assign(phiHbyA - pEqn.flux())

        U.assign(HbyA + rAU * fvc.reconstruct((phig - pEqn.flux()) / rAUf))
        U.correctBoundaryConditions()
        p.assign(p_rgh + rhok * gh)

        sum_local, global_err = pyf.computeContinuityErrors(phi)
        cumulativeContErr[0] += global_err
        pyf.Info(
            f"time step continuity errors : sum local = {sum_local}, "
            f"global = {global_err}, cumulative = {cumulativeContErr[0]}"
        )


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


def inner_loop(ctx: Any) -> bool:
    return bool(ctx.models["pimple_control"].loop())


def momentum(
    model_state: Any,
    U: Any,
    phi: Any,
    p: Any,
    turbulence: Any,
    pimple_control: Any,
    p_rgh: Any = None,
    rhok: Any = None,
    ghf: Any = None,
) -> FieldUpdates:
    if model_state.use_boussinesq:
        if p_rgh is None or rhok is None or ghf is None:
            raise RuntimeError("Boussinesq mode requires p_rgh, rhok, and ghf fields")
        ensure_pressure_reference(model_state, p, U.mesh(), p_rgh)
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
        ensure_pressure_reference(model_state, p, U.mesh())
        UEqn = momentum_helper(
            U=U,
            phi=phi,
            p=p,
            turbulence=turbulence,
            pimple_control=pimple_control,
        )

    return FieldUpdates({"UEqn": UEqn, "U": U})


def continuity(
    model_state: Any,
    U: Any,
    p: Any,
    phi: Any,
    UEqn: Any,
    pimple_control: Any,
    cumulativeContErr: list[float],
    p_rgh: Any = None,
    rhok: Any = None,
    gh: Any = None,
    ghf: Any = None,
) -> FieldUpdates:
    if model_state.use_boussinesq:
        if p_rgh is None or rhok is None or gh is None or ghf is None:
            raise RuntimeError(
                "Boussinesq mode requires p_rgh, rhok, gh, and ghf fields"
            )
        ensure_pressure_reference(model_state, p, U.mesh(), p_rgh)
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
            pRefCell=model_state.pRefCell,
            pRefValue=model_state.pRefValue,
        )
        return FieldUpdates({"U": U, "p": p, "phi": phi, "p_rgh": p_rgh})

    ensure_pressure_reference(model_state, p, U.mesh())
    continuity_helper(
        U=U,
        p=p,
        phi=phi,
        UEqn=UEqn,
        pimple_control=pimple_control,
        cumulativeContErr=cumulativeContErr,
        pRefCell=model_state.pRefCell,
        pRefValue=model_state.pRefValue,
    )
    return FieldUpdates({"U": U, "p": p, "phi": phi})
