# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from typing import Annotated, Any, Callable, Optional, Protocol

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
from neofoam.algorithms.control import PimpleControl
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization import field, model
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata
from .control_factory import create_pimple_control

from ..incompressibleFluidModel import Model

pimple = Model("Pimple")


class PressureReferenceState(Protocol):
    algorithm_type: str
    use_boussinesq: bool


class TurbulenceModel(Protocol):
    def divDevReff(self, velocity: volVectorField) -> object: ...


@pimple.build
def build(self: Any) -> list[object]:
    def create_phi(context: dict[str, object]) -> surfaceScalarField:
        return pyf.createPhi(context["fields.U"])

    def create_cumulative_cont_err(_context: dict[str, object]) -> list[float]:
        return [0.0]

    def create_pressure_reference(context: dict[str, object]) -> dict[str, object]:
        """Initialize pressure reference cell and value."""
        p = context["fields.p"]
        mesh = context["mesh"]
        p_rgh = context.get("fields.p_rgh") if pimple.use_boussinesq else None

        fv_solution = pyf.dictionary.read("system/fvSolution")
        algo_dict = fv_solution.subDict(pimple.algorithm_type)
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

        mesh.setFluxRequired(pyf.Word("p"))
        if p_rgh is not None:
            mesh.setFluxRequired(pyf.Word("p_rgh"))

        return {"pRefCell": pRefCell, "pRefValue": pRefValue}

    init_steps = [
        read_vol_field(volScalarField, "p"),
        read_vol_field(volVectorField, "U"),
        field("phi", create_phi, depends_on=["fields.U"]),
        model("pimple_control", create_pimple_control, depends_on=["mesh"]),
        model("cumulativeContErr", create_cumulative_cont_err),
    ]

    # Add pressure reference model - depends on p_rgh if boussinesq is enabled
    # Note: p_rgh field is created by the boussinesq model, not here
    pressure_ref_deps = ["fields.p", "mesh"]
    if pimple.use_boussinesq:
        pressure_ref_deps.append("fields.p_rgh")
    init_steps.append(
        model(
            "pressure_reference",
            create_pressure_reference,
            depends_on=pressure_ref_deps,
        )
    )

    return init_steps


def inner_loop(ctx: Context) -> bool:
    return bool(ctx.models["pimple_control"].loop(ctx))


def _alias_operation(
    op_func: Callable[..., FieldUpdates],
    *,
    operation_name: str,
    depends_on: list[str],
) -> Operation:
    return Operation(
        func=SequentialOp(op_func),
        metadata=OperationMetadata(
            op_name=operation_name,
            depends_on=depends_on,
            shape="box",
            color="lightblue",
        ),
    )


@pimple.operation(operation_number="2.1")
def momentum(
    U: volVectorField,
    phi: surfaceScalarField,
    p: volScalarField,
    turbulence: Annotated[TurbulenceModel, "models"],
    pimple_control: Annotated[PimpleControl, "models"],
) -> FieldUpdates:
    UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))
    UEqn.relax()

    if pimple_control.momentumPredictor():
        pyf.solve(UEqn + fvc.grad(p))

    return FieldUpdates({"UEqn": UEqn, "U": U})


@pimple.operation(operation_number="2.2", depends_on=["momentum"])
def continuity(
    U: volVectorField,
    p: volScalarField,
    phi: surfaceScalarField,
    UEqn: fvVectorMatrix,
    pimple_control: Annotated[PimpleControl, "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    pressure_reference: Annotated[dict[str, object], "models"],
) -> FieldUpdates:
    pRefCell = pressure_reference["pRefCell"]
    pRefValue = pressure_reference["pRefValue"]

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

    return FieldUpdates({"U": U, "p": p, "phi": phi})


@pimple.operation(operation_number="2.1")
def momentum_boussinesq(
    U: volVectorField,
    phi: surfaceScalarField,
    p: volScalarField,
    turbulence: Annotated[TurbulenceModel, "models"],
    pimple_control: Annotated[PimpleControl, "models"],
    p_rgh: volScalarField,
    rhok: volScalarField,
    ghf: surfaceScalarField,
) -> FieldUpdates:
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

    return FieldUpdates({"UEqn": UEqn, "U": U})


@pimple.operation(operation_number="2.2", depends_on=["momentum_boussinesq"])
def continuity_boussinesq(
    U: volVectorField,
    p: volScalarField,
    phi: surfaceScalarField,
    UEqn: fvVectorMatrix,
    pimple_control: Annotated[PimpleControl, "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    pressure_reference: Annotated[dict[str, object], "models"],
    p_rgh: volScalarField,
    rhok: volScalarField,
    gh: volScalarField,
    ghf: surfaceScalarField,
) -> FieldUpdates:
    pRefCell = pressure_reference["pRefCell"]
    pRefValue = pressure_reference["pRefValue"]
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

    return FieldUpdates({"U": U, "p": p, "phi": phi, "p_rgh": p_rgh})


@pimple.operation_collection
def collected_operations(self: Any) -> Operations:
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=IterativeOp(inner_loop),
            metadata=OperationMetadata(op_name="inner_loop"),
        )
    )

    if getattr(self, "use_boussinesq", False):
        momentum_op = momentum_boussinesq
        continuity_op = continuity_boussinesq
    else:
        momentum_op = momentum
        continuity_op = continuity

    model_ops.add(
        _alias_operation(momentum_op, operation_name="momentum", depends_on=[])
    )
    model_ops.add(
        _alias_operation(
            continuity_op, operation_name="continuity", depends_on=["momentum"]
        )
    )
    return model_ops
