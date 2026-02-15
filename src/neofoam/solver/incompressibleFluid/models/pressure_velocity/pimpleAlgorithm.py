# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

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
from .control_factory import create_pimple_control

from ..incompressibleFluidModel import Model

pimple = Model("Pimple")
_active_model_state: Any = None


def _set_active_model_state(model_state: Any) -> None:
    global _active_model_state
    _active_model_state = model_state


def _get_active_model_state() -> Any:
    if _active_model_state is None:
        raise RuntimeError(
            "PIMPLE model state not configured before operation dispatch"
        )
    return _active_model_state


def ensure_pressure_reference(
    model_state: Any, p: Any, mesh: Any, p_rgh: Any = None
) -> None:
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


@pimple.build
def build() -> list[Any]:
    def create_phi(context: dict[str, Any]) -> Any:
        return pyf.createPhi(context["fields.U"])

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
    return bool(ctx.models["pimple_control"].loop(ctx))


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


@pimple.operation(operation_number="2.1")
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
    model_state = _get_active_model_state()
    ensure_pressure_reference(model_state, p, U.mesh())

    UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))
    UEqn.relax()

    if pimple_control.momentumPredictor():
        pyf.solve(UEqn + fvc.grad(p))

    return FieldUpdates({"UEqn": UEqn, "U": U})


@pimple.operation(operation_number="2.2", depends_on=["momentum"])
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
    model_state = _get_active_model_state()
    ensure_pressure_reference(model_state, p, U.mesh())

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
            pEqn.setReference(model_state.pRefCell, model_state.pRefValue, False)
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
    U: Any,
    phi: Any,
    p: Any,
    turbulence: Annotated[Any, "models"],
    pimple_control: Annotated[Any, "models"],
    p_rgh: Any = None,
    rhok: Any = None,
    ghf: Any = None,
) -> FieldUpdates:
    model_state = _get_active_model_state()
    if p_rgh is None or rhok is None or ghf is None:
        raise RuntimeError("Boussinesq mode requires p_rgh, rhok, and ghf fields")

    ensure_pressure_reference(model_state, p, U.mesh(), p_rgh)
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
    model_state = _get_active_model_state()
    if p_rgh is None or rhok is None or gh is None or ghf is None:
        raise RuntimeError("Boussinesq mode requires p_rgh, rhok, gh, and ghf fields")

    ensure_pressure_reference(model_state, p, U.mesh(), p_rgh)
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
            pEqn.setReference(model_state.pRefCell, model_state.pRefValue, False)
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

    if model_state.use_boussinesq:
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
