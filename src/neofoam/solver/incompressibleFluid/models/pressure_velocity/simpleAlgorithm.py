# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""SIMPLE pressure-velocity algorithm model."""

from typing import Annotated, Any, Callable, Optional, Protocol

import pybFoam as pyf
from pybFoam import (
    fvc,
    fvm,
    fvMesh,
    fvScalarMatrix,
    fvVectorMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)

from neofoam.foam.initialization import read_vol_field
from neofoam.algorithms.control import SimpleControl
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization import field, model
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata
from .control_factory import create_simple_control

from ..incompressibleFluidModel import Model

simple = Model("Simple")


class PressureReferenceState(Protocol):
    pRefCell: Optional[object]
    pRefValue: Optional[object]
    fv_solution: Optional[object]
    algorithm_type: str


class TurbulenceModel(Protocol):
    def divDevReff(self, velocity: volVectorField) -> object: ...


_active_model_state: Optional[PressureReferenceState] = None


def _set_active_model_state(model_state: PressureReferenceState) -> None:
    global _active_model_state
    _active_model_state = model_state


def _get_active_model_state() -> PressureReferenceState:
    if _active_model_state is None:
        raise RuntimeError(
            "SIMPLE model state not configured before operation dispatch"
        )
    return _active_model_state


def ensure_pressure_reference(
    model_state: PressureReferenceState,
    p: volScalarField,
    mesh: fvMesh,
) -> None:
    if model_state.pRefCell is not None and model_state.pRefValue is not None:
        return

    if model_state.fv_solution is None:
        model_state.fv_solution = pyf.dictionary.read("system/fvSolution")

    algo_dict = model_state.fv_solution.subDict(model_state.algorithm_type)
    pRefCell, pRefValue = pyf.setRefCell(p, algo_dict)

    model_state.pRefCell = pRefCell
    model_state.pRefValue = pRefValue
    mesh.setFluxRequired(pyf.Word("p"))


@simple.build
def build(self: Any) -> list[object]:
    def create_phi(context: dict[str, object]) -> surfaceScalarField:
        return pyf.createPhi(context["fields.U"])

    def create_cumulative_cont_err(_context: dict[str, object]) -> list[float]:
        return [0.0]

    return [
        read_vol_field(volScalarField, "p"),
        read_vol_field(volVectorField, "U"),
        field("phi", create_phi, depends_on=["fields.U"]),
        model("simple_control", create_simple_control, depends_on=["mesh"]),
        model("cumulativeContErr", create_cumulative_cont_err),
    ]


def inner_loop(ctx: Context) -> bool:
    return bool(ctx.models["simple_control"].loop(ctx))


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


@simple.operation(operation_number="2.1")
def momentum(
    U: volVectorField,
    phi: surfaceScalarField,
    p: volScalarField,
    turbulence: Annotated[TurbulenceModel, "models"],
    simple_control: Annotated[SimpleControl, "models"],
) -> FieldUpdates:
    model_state = _get_active_model_state()
    ensure_pressure_reference(model_state, p, U.mesh())

    UEqn = fvVectorMatrix(fvm.div(phi, U) + turbulence.divDevReff(U))
    UEqn.relax()

    if simple_control.momentumPredictor():
        pyf.solve(UEqn + fvc.grad(p))

    return FieldUpdates({"UEqn": UEqn, "U": U})


@simple.operation(operation_number="2.2", depends_on=["momentum"])
def continuity(
    U: volVectorField,
    p: volScalarField,
    phi: surfaceScalarField,
    UEqn: fvVectorMatrix,
    simple_control: Annotated[SimpleControl, "models"],
    cumulativeContErr: Annotated[list[float], "models"],
) -> FieldUpdates:
    model_state = _get_active_model_state()
    ensure_pressure_reference(model_state, p, U.mesh())

    rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
    HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p))
    phiHbyA = surfaceScalarField(pyf.Word("phiHbyA"), fvc.flux(HbyA))

    pyf.adjustPhi(phiHbyA, U, p)

    rAtU = volScalarField(rAU)
    use_consistent = (
        hasattr(simple_control, "consistent") and simple_control.consistent()
    )
    if use_consistent:
        rAtU.assign(1.0 / (1.0 / rAU - UEqn.H1()))
        phiHbyA.assign(
            phiHbyA + fvc.interpolate(rAtU - rAU) * fvc.snGrad(p) * U.mesh().magSf()
        )
        HbyA.assign(HbyA - (rAU - rAtU) * fvc.grad(p))

    pyf.constrainPressure(p, U, phiHbyA, rAtU)

    while simple_control.correctNonOrthogonal():
        pEqn = fvScalarMatrix(fvm.laplacian(rAtU, p) - fvc.div(phiHbyA))
        pEqn.setReference(model_state.pRefCell, model_state.pRefValue, False)
        pEqn.solve()

        if simple_control.finalNonOrthogonalIter():
            phi.assign(phiHbyA - pEqn.flux())

    sum_local, global_err = pyf.computeContinuityErrors(phi)
    cumulativeContErr[0] += global_err
    pyf.Info(
        f"time step continuity errors : sum local = {sum_local}, "
        f"global = {global_err}, cumulative = {cumulativeContErr[0]}"
    )

    p.relax()
    U.assign(HbyA - rAtU * fvc.grad(p))
    U.correctBoundaryConditions()

    return FieldUpdates({"U": U, "p": p, "phi": phi})


@simple.operation_collection
def collected_operations(self: Any) -> Operations:
    _set_active_model_state(self)
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=IterativeOp(inner_loop),
            metadata=OperationMetadata(op_name="inner_loop"),
        )
    )
    model_ops.add(_alias_operation(momentum, operation_name="momentum", depends_on=[]))
    model_ops.add(
        _alias_operation(
            continuity, operation_name="continuity", depends_on=["momentum"]
        )
    )
    return model_ops
