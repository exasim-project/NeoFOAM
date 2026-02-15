# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""SIMPLE pressure-velocity algorithm model."""

from typing import Annotated, Any

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
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
)
from .control_factory import create_simple_control

from ..incompressibleFluidModel import Model

simple = Model("Simple")
_active_model_state: Any = None


def _set_active_model_state(model_state: Any) -> None:
    global _active_model_state
    _active_model_state = model_state


def _get_active_model_state() -> Any:
    if _active_model_state is None:
        raise RuntimeError(
            "SIMPLE model state not configured before operation dispatch"
        )
    return _active_model_state


def ensure_pressure_reference(model_state: Any, p: Any, mesh: Any) -> None:
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
def build() -> list[Any]:
    def create_phi(context: dict[str, Any]) -> Any:
        return pyf.createPhi(context["fields.U"])

    def create_cumulative_cont_err(_context: dict[str, Any]) -> list[float]:
        return [0.0]

    return [
        read_vol_field(volScalarField, "p"),
        read_vol_field(volVectorField, "U"),
        field("phi", create_phi, depends_on=["fields.U"]),
        model("simple_control", create_simple_control, depends_on=["mesh"]),
        model("cumulativeContErr", create_cumulative_cont_err),
    ]


def inner_loop(ctx: Any) -> bool:
    return bool(ctx.models["simple_control"].loop(ctx))


def _alias_operation(
    op_func: Any,
    *,
    operation_name: str,
    depends_on: list[str],
) -> Operation:
    metadata = getattr(op_func, "_metadata", None)
    return Operation(
        func=SequentialOp(op_func),
        operation_number=getattr(metadata, "operation_number", None),
        operation_name=operation_name,
        domain_name=None,
        depends_on=depends_on,
        before=[],
        shape="box",
        color="lightblue",
        level=0,
    )


@simple.operation(operation_number="2.1")
def momentum(
    U: Any,
    phi: Any,
    p: Any,
    turbulence: Annotated[Any, "models"],
    simple_control: Annotated[Any, "models"],
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
    U: Any,
    p: Any,
    phi: Any,
    UEqn: Any,
    simple_control: Annotated[Any, "models"],
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
def collected_operations(self, model_state: Any) -> Operations:
    _set_active_model_state(model_state)
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=IterativeOp(inner_loop),
            operation_name="inner_loop",
            operation_number=None,
        )
    )
    model_ops.add(_alias_operation(momentum, operation_name="momentum", depends_on=[]))
    model_ops.add(
        _alias_operation(
            continuity, operation_name="continuity", depends_on=["momentum"]
        )
    )
    return model_ops
