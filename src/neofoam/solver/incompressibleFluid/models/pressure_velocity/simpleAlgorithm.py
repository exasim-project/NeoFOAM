# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""SIMPLE pressure-velocity coupling (steady state).

Port of ``simpleFoam``'s loop body (``UEqn.H`` / ``pEqn.H``) to the
SolverSpec/ModelSpec API, adapted from the ``feat/incompressibleFluid``
source branch. The momentum equation carries no ddt term (the case's
``ddtSchemes`` is expected to be ``steadyState``), equations are
under-relaxed via ``relax()``, and the pressure field is explicitly
relaxed after each corrector (``p.relax()``), exactly as in simpleFoam.

One solver "time step" is one SIMPLE outer iteration: the inner loop
predicate (``SimpleControl.loop``) admits a single momentum+continuity
pass per step and the turbulence model is corrected once per step by the
solver graph, mirroring ``simpleFoam``'s ``while (simple.loop())`` body.
"""

from typing import Annotated, Any, Callable, Protocol

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

from neofoam import telemetry
from neofoam.fields import (
    CalculatedBC,
    CyclicBC,
    EmptyBC,
    FixedValueBC,
    GenericBC,
    InletOutletBC,
    NoSlipBC,
    PressureInletOutletVelocityBC,
    Scalar,
    SlipBC,
    SymmetryBC,
    SymmetryPlaneBC,
    Vector,
    ZeroGradientBC,
)
from neofoam.foam import fvSchemes, fvSolution
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.dependency_resolver import wrap_with_dependency_resolution
from neofoam.framework.initialization import field, model
from neofoam.framework.model import BoundExtension
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata

from ..incompressibleFluidModel import Model
from .control_factory import create_simple_control
from .extension import momentum_extension, pressure_extension

simple = Model("Simple")

# Per-spec fvSchemes / fvSolution slices (see pimpleAlgorithm).
SimpleFvSchemes = simple.config(fvSchemes)
SimpleFvSolution = simple.config(fvSolution)

# Optional SIMPLE control keys read straight from ``system/fvSolution`` by
# ``setRefCell`` — a closed domain needs a pressure reference.
SimpleFvSolution.add_controls("SIMPLE", pRefCell=int, pRefValue=float)

# 0/<name> field declarations SIMPLE owns (same arm sets as PIMPLE).
simple.field(
    "U",
    dimensions=[0, 1, -1, 0, 0, 0, 0],
    value_type=Vector,
    allowed_bcs=[
        NoSlipBC,
        FixedValueBC,
        ZeroGradientBC,
        SlipBC,
        InletOutletBC,
        PressureInletOutletVelocityBC,
        EmptyBC,
        SymmetryBC,
        SymmetryPlaneBC,
        CyclicBC,
        GenericBC,
    ],
    write=True,
)
simple.field(
    "p",
    dimensions=[0, 2, -2, 0, 0, 0, 0],
    value_type=Scalar,
    allowed_bcs=[
        FixedValueBC,
        ZeroGradientBC,
        InletOutletBC,
        CalculatedBC,
        EmptyBC,
        SymmetryBC,
        SymmetryPlaneBC,
        CyclicBC,
        GenericBC,
    ],
    write=True,
)


class ViscousStress(Protocol):
    def update(self, ctx: Context) -> None: ...
    def divDevReff(self, U: volVectorField) -> Any: ...


@simple.build
def build(self: Any) -> list[Any]:
    """Lazy initializers for SIMPLE state (non-field bits only).

    ``U`` / ``p`` are auto-synthesized from the ``simple.field(...)``
    declarations; ``@build`` carries ``phi``, the SimpleControl object,
    the running continuity-error accumulator, and the pressure-reference
    cell logic (reading the ``SIMPLE`` subdict of ``system/fvSolution``).
    """

    def create_phi(context: dict[str, Any]) -> surfaceScalarField:
        return pyf.createPhi(context["fields.U"])

    def create_cumulative_cont_err(_context: dict[str, Any]) -> list[float]:
        return [0.0]

    def create_pressure_reference(context: dict[str, Any]) -> dict[str, Any]:
        p = context["fields.p"]
        mesh = context["mesh"]

        fv_solution = pyf.dictionary.read("system/fvSolution")
        algo_dict = fv_solution.subDict("SIMPLE")
        pRefCell, pRefValue = pyf.setRefCell(p, algo_dict)

        mesh.setFluxRequired(pyf.Word("p"))
        return {"pRefCell": pRefCell, "pRefValue": pRefValue}

    return [
        field("phi", create_phi, depends_on=["fields.U"], write=True),
        model("simple_control", create_simple_control, depends_on=["mesh"]),
        model("cumulativeContErr", create_cumulative_cont_err),
        model(
            "pressure_reference",
            create_pressure_reference,
            depends_on=["fields.p", "mesh"],
        ),
    ]


def inner_loop(ctx: Context) -> bool:
    """One momentum+continuity pass per solver step (``SimpleControl.loop``)."""
    return bool(ctx.models["simple_control"].loop(ctx))


@simple.operation(operation_number="2.1")
@SimpleFvSchemes.add(
    ddt="ddt(U)",
    div=["div(phi,U)", "div((nuEff*dev2(T(grad(U)))))"],
    grad="grad(U)",
    laplacian="laplacian(nuEff,U)",
)
@SimpleFvSolution.add("U")
def momentum(
    U: volVectorField,
    phi: surfaceScalarField,
    p: volScalarField,
    viscousStress: Annotated[ViscousStress, "models"],
    simple_control: Annotated[Any, "models"],
    ctx: Context,
    ext: Annotated[BoundExtension, momentum_extension],
) -> FieldUpdates:
    # Start-of-iteration prevIter snapshot: ``simpleControl.loop()`` calls
    # ``storePrevIterFields()`` natively so that ``p.relax()`` (explicit field
    # under-relaxation in the pressure corrector) has a previous state; p is
    # untouched until the pressure solve, so storing here is equivalent.
    p.storePrevIter()

    # Refresh nuEff where it is consumed (see pimpleAlgorithm.momentum).
    with telemetry.span("momentum.assemble"):
        viscousStress.update(ctx)
        # UEqn.H: the wall velocities on the MRF patches are set first (they
        # feed the boundary coefficients of ``div(phi,U)``); then every active
        # model's terms fold into the sum in registration order — MRF's frame
        # acceleration with ``+``, the fvOptions source with ``-`` (native's
        # ``== fvOptions(U)``). One term site, so both join after the viscous
        # stress, where native adds DDt(U) before it.
        ext.correct_boundary_velocity(U)
        UEqn = fvVectorMatrix(fvm.div(phi, U) + viscousStress.divDevReff(U) + ext.terms(U))
        # The source joined the sum BEFORE relaxation; the constraints apply
        # AFTER it, and the correction runs after the solve.
        UEqn.relax()
        ext.constrain(UEqn)

    if simple_control.momentumPredictor():
        with telemetry.span("momentum.solve"):
            fvVectorMatrix(UEqn + fvc.grad(p)).solve()
        ext.correct(U)

    return FieldUpdates({"UEqn": UEqn, "U": U})


@simple.operation(operation_number="2.2", depends_on=["momentum"])
@SimpleFvSchemes.add(
    grad="grad(p)",
    laplacian="laplacian(rAtU,p)",
    interpolation=["flux(HbyA)", "interpolate((1|A(U)))"],
    snGrad="snGrad(p)",
)
@SimpleFvSolution.add("p")
def continuity(
    U: volVectorField,
    p: volScalarField,
    phi: surfaceScalarField,
    UEqn: fvVectorMatrix,
    simple_control: Annotated[Any, "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    pressure_reference: Annotated[dict[str, Any], "models"],
    ext: Annotated[BoundExtension, pressure_extension],
) -> FieldUpdates:
    pRefCell = pressure_reference["pRefCell"]
    pRefValue = pressure_reference["pRefValue"]

    with telemetry.span("pressure.flux"):
        rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
        HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p))
        phiHbyA = surfaceScalarField(pyf.Word("phiHbyA"), fvc.flux(HbyA))

        # pEqn.H hands the pressure equation the flux seen from the rotating
        # frame; ``adjustPhi`` then balances that relative flux.
        ext.make_relative(phiHbyA)

        pyf.adjustPhi(phiHbyA, U, p)

        # SIMPLEC (``consistent yes``): sharpen the pressure-laplacian
        # coefficient with the H1 diagonal contribution.
        rAtU = volScalarField(rAU)
        if simple_control.consistent():
            rAtU.assign(1.0 / (1.0 / rAU - UEqn.H1()))
            phiHbyA.assign(phiHbyA + fvc.interpolate(rAtU - rAU) * fvc.snGrad(p) * U.mesh().magSf())
            HbyA.assign(HbyA - (rAU - rAtU) * fvc.grad(p))

        if not ext.constrain_pressure(p, U, phiHbyA, rAtU):
            pyf.constrainPressure(p, U, phiHbyA, rAtU)

    while simple_control.correctNonOrthogonal():
        with telemetry.span("pressure.assemble"):
            pEqn = fvScalarMatrix(fvm.laplacian(rAtU, p) - fvc.div(phiHbyA))
            pEqn.setReference(pRefCell, pRefValue, False)
        with telemetry.span("pressure.solve"):
            pEqn.solve()

        if simple_control.finalNonOrthogonalIter():
            phi.assign(phiHbyA - pEqn.flux())

    sum_local, global_err = pyf.computeContinuityErrors(phi)
    cumulativeContErr[0] += global_err
    pyf.Info(
        f"time step continuity errors : sum local = {sum_local}, "
        f"global = {global_err}, cumulative = {cumulativeContErr[0]}"
    )

    # Explicit pressure under-relaxation (relaxationFactors.fields), then the
    # velocity correction — the simpleFoam ordering.
    p.relax()
    U.assign(HbyA - rAtU * fvc.grad(p))
    U.correctBoundaryConditions()
    # pEqn.H closes on a second ``fvOptions.correct(U)``: the corrector has
    # just overwritten U, so any correction the predictor applied is gone.
    ext.correct(U)

    return FieldUpdates({"U": U, "p": p, "phi": phi})


def _alias_operation(
    op_func: Callable[..., Any],
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


@simple.operation_collection
def collected_operations(self: Any) -> Operations:
    """Wrap momentum/continuity inside the one-pass SIMPLE inner loop."""
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=IterativeOp(inner_loop),
            metadata=OperationMetadata(op_name="inner_loop"),
        )
    )

    wrapped_momentum = wrap_with_dependency_resolution(momentum, self, simple._dependency_resolver)
    wrapped_continuity = wrap_with_dependency_resolution(
        continuity, self, simple._dependency_resolver
    )

    model_ops.add(_alias_operation(wrapped_momentum, operation_name="momentum", depends_on=[]))
    model_ops.add(
        _alias_operation(
            wrapped_continuity,
            operation_name="continuity",
            depends_on=["momentum"],
        )
    )
    return model_ops
