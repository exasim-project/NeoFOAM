# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from typing import Any, Literal

import pybFoam as pyf  # type: ignore[import-not-found]
from pybFoam import (
    Info,
    fvc,
    fvm,
    fvScalarMatrix,
    fvVectorMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)
from pybFoam.turbulence import incompressibleTurbulenceModel, singlePhaseTransportModel  # type: ignore[import-not-found]
from pydantic import BaseModel

from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    StepBuilder,
)
from foamadapter.framework.solver import Solver


class CFLCondition:
    """Condition for CFL-based time stepping."""

    def __init__(self, maxDeltaT: float) -> None:
        self.GREAT = 1e30
        self.SMALL = 1e-15
        controlDict = pyf.dictionary.read("system/controlDict")
        self.adjustable = controlDict.get[bool]("adjustTimeStep")
        self.maxCFL = controlDict.get[float]("maxCo")
        self.maxDeltaT = maxDeltaT
        self.maxRatio = 1.2

    def __call__(self, ctx: Context) -> bool:
        """Adjust time step based on CFL number and continue."""
        runTime = ctx.runTime
        phi = ctx.fields["phi"]

        if not self.adjustable:
            runTime.increment()
            return runTime.loop()

        deltaT = runTime.deltaTValue()
        maxCFLNumber, meanCFLNumber = pyf.computeCFLNumber(phi)
        Info(f"Courant Number mean: {meanCFLNumber}, max: {maxCFLNumber}")

        ratio = self.maxCFL / maxCFLNumber if maxCFLNumber > self.SMALL else self.GREAT
        ratio = min(ratio, self.maxRatio)

        if ratio > self.SMALL and ratio < self.GREAT:
            finalDeltaT = min(deltaT * ratio, self.maxDeltaT)
            runTime.setDeltaT(finalDeltaT)

        runTime.increment()
        return runTime.loop()


class PimpleLoopCondition:
    """Condition for PIMPLE iterations."""

    def __init__(self) -> None:
        self.pimple = None

    def __call__(self, ctx: Context) -> bool:
        if self.pimple is None:
            self.pimple = ctx.models.get("pimple")
            if self.pimple is None:
                raise ValueError("pimple control not found in context")
        return self.pimple.loop()


class PimpleCorrectorCondition:
    """Condition for PIMPLE pressure correctors."""

    def __init__(self) -> None:
        self.pimple = None

    def __call__(self, ctx: Context) -> bool:
        if self.pimple is None:
            self.pimple = ctx.models.get("pimple")
            if self.pimple is None:
                raise ValueError("pimple control not found in context")
        return self.pimple.correct()


class NonOrthogonalCondition:
    """Condition for non-orthogonal corrections."""

    def __init__(self) -> None:
        self.pimple = None

    def __call__(self, ctx: Context) -> bool:
        if self.pimple is None:
            self.pimple = ctx.models.get("pimple")
            if self.pimple is None:
                raise ValueError("pimple control not found in context")
        return self.pimple.correctNonOrthogonal()


@Solver
class IncompressibleFluid(BaseModel):
    """
    Incompressible fluid solver using the PIMPLE algorithm.
    This is a framework-based port of pimplefoam.py
    """

    name: Literal["IncompressibleFluid"] = "IncompressibleFluid"
    argv: list[str] = []
    pRefCell: int | None = None
    pRefValue: float | None = None
    maxDeltaT: float = 1e5

    def create_context(self) -> Context:
        """Initialize the simulation context with mesh, runtime, and fields."""
        argList = pyf.argList(self.argv)
        runTime = pyf.Time(argList)
        mesh = pyf.fvMesh(runTime)

        # Read controlDict
        controlDict = pyf.dictionary.read("system/controlDict")
        try:
            self.maxDeltaT = controlDict.get[float]("maxDeltaT")
        except KeyError:
            pass

        # Create context
        ctx = Context(fields={}, models={}, mesh=mesh, runTime=runTime)

        return ctx

    @Solver.operation(operation_number=1)
    def create_fields(self, ctx: Context) -> FieldUpdates:
        """Create and read fields from disk."""
        mesh = ctx.mesh

        p = volScalarField.read_field(mesh, "p")
        U = volVectorField.read_field(mesh, "U")
        phi = pyf.createPhi(U)

        laminarTransport = singlePhaseTransportModel(U, phi)
        turbulence = incompressibleTurbulenceModel.New(U, phi, laminarTransport)

        # Read fvSolution and set reference cell
        fvSolution = pyf.dictionary.read("system/fvSolution")
        self.pRefCell, self.pRefValue = pyf.setRefCell(p, fvSolution.subDict("PIMPLE"))
        mesh.setFluxRequired(pyf.Word("p"))

        # Create pimple control
        pimple = pyf.pimpleControl(mesh)

        return FieldUpdates(
            {
                "p": p,
                "U": U,
                "phi": phi,
                "laminarTransport": laminarTransport,
                "turbulence": turbulence,
                "pimple": pimple,
            }
        )

    @Solver.operation(operation_number=2, depends_on=["create_fields"])
    def setup_models(self, ctx: Context) -> None:
        """Move pimple control to models for easy access."""
        pimple = ctx.fields.pop("pimple")
        ctx.models["pimple"] = pimple

    @Solver.operation(operation_number=3, depends_on=["setup_models"])
    def print_time(self, ctx: Context) -> None:
        """Print current simulation time."""
        runTime = ctx.runTime
        Info(f"Time = {runTime.timeName()}")

    @Solver.operation(operation_number=4, depends_on=["print_time"])
    def momentum_predictor(
        self, U: volVectorField, phi: surfaceScalarField, turbulence: Any
    ) -> FieldUpdates:
        """Solve the momentum equation."""
        UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))
        UEqn.relax()

        return FieldUpdates({"UEqn": UEqn})

    @Solver.operation(operation_number=5, depends_on=["momentum_predictor"])
    def solve_momentum(
        self, ctx: Context, p: volScalarField, UEqn: fvVectorMatrix
    ) -> None:
        """Solve momentum equation if momentum predictor is enabled."""
        pimple = ctx.models["pimple"]
        if pimple.momentumPredictor():
            pyf.solve(UEqn + fvc.grad(p))

    @Solver.operation(operation_number=6, depends_on=["solve_momentum"])
    def compute_HbyA(
        self, U: volVectorField, p: volScalarField, UEqn: fvVectorMatrix
    ) -> FieldUpdates:
        """Compute H/A for pressure equation."""
        rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
        HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p))

        return FieldUpdates({"rAU": rAU, "HbyA": HbyA})

    @Solver.operation(operation_number=7, depends_on=["compute_HbyA"])
    def compute_phiHbyA(
        self,
        U: volVectorField,
        phi: surfaceScalarField,
        HbyA: volVectorField,
        rAU: volScalarField,
    ) -> FieldUpdates:
        """Compute flux from H/A."""
        phiHbyA = surfaceScalarField(
            pyf.Word("phiHbyA"),
            fvc.flux(HbyA) + fvc.interpolate(rAU) * fvc.ddtCorr(U, phi),
        )
        return FieldUpdates({"phiHbyA": phiHbyA})

    @Solver.operation(operation_number=8, depends_on=["compute_phiHbyA"])
    def adjust_phi(
        self,
        U: volVectorField,
        p: volScalarField,
        phiHbyA: surfaceScalarField,
        rAU: volScalarField,
    ) -> None:
        """Adjust flux for continuity."""
        pyf.adjustPhi(phiHbyA, U, p)
        pyf.constrainPressure(p, U, phiHbyA, rAU)

    @Solver.operation(operation_number=9, depends_on=["adjust_phi"])
    def solve_pressure(
        self,
        ctx: Context,
        p: volScalarField,
        rAU: volScalarField,
        phiHbyA: surfaceScalarField,
    ) -> None:
        """Solve pressure equation."""
        pimple = ctx.models["pimple"]
        pEqn = fvScalarMatrix(fvm.laplacian(rAU, p) - fvc.div(phiHbyA))
        pEqn.setReference(self.pRefCell, self.pRefValue, False)
        pEqn.solve(p.select(pimple.finalInnerIter()))

        ctx.fields["pEqn"] = pEqn

    @Solver.operation(operation_number=10, depends_on=["solve_pressure"])
    def update_flux(
        self,
        ctx: Context,
        phi: surfaceScalarField,
        phiHbyA: surfaceScalarField,
        pEqn: fvScalarMatrix,
    ) -> None:
        """Update flux after pressure solution."""
        pimple = ctx.models["pimple"]
        if pimple.finalNonOrthogonalIter():
            phi.assign(phiHbyA - pEqn.flux())

    @Solver.operation(operation_number=11, depends_on=["update_flux"])
    def correct_velocity(
        self,
        U: volVectorField,
        p: volScalarField,
        HbyA: volVectorField,
        rAU: volScalarField,
    ) -> None:
        """Correct velocity field."""
        U.assign(HbyA - rAU * fvc.grad(p))
        U.correctBoundaryConditions()

    @Solver.operation(operation_number=12, depends_on=["correct_velocity"])
    def turbulence_correction(
        self, ctx: Context, laminarTransport: Any, turbulence: Any
    ) -> None:
        """Correct turbulence model if needed."""
        pimple = ctx.models["pimple"]
        if pimple.turbCorr():
            laminarTransport.correct()
            turbulence.correct()

    @Solver.operation(operation_number=13, depends_on=["turbulence_correction"])
    def write_output(self, ctx: Context) -> None:
        """Write fields to disk."""
        runTime = ctx.runTime
        runTime.write(True)
        runTime.printExecutionTime()

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        """Collect all decorated operations."""
        _ = domain_name  # Part of SolverInterface, unused in this implementation
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops

    def main_loop(self, ctx: Context) -> None:
        """
        Main simulation loop with PIMPLE algorithm structure:
        - Time loop
          - PIMPLE loop
            - Momentum predictor
            - Pressure corrector loop
              - Non-orthogonal corrector loop
        """
        ops = self.operations()

        # Build the main execution graph
        main_loop = StepBuilder()

        # Time loop
        time_loop_op = Operation(
            func=IterativeOp(CFLCondition(self.maxDeltaT)),
            operation_name="time_loop",
            operation_number=1,
        )

        with main_loop.loop(time_loop_op) as time_loop:
            time_loop.step(ops["print_time"])

            # PIMPLE outer loop
            pimple_loop_op = Operation(
                func=IterativeOp(PimpleLoopCondition()),
                operation_name="pimple_loop",
                operation_number=2,
            )

            with time_loop.loop(pimple_loop_op) as pimple_loop:
                # Momentum predictor
                pimple_loop.step(ops["momentum_predictor"])
                pimple_loop.step(ops["solve_momentum"])

                # Pressure corrector loop
                corrector_loop_op = Operation(
                    func=IterativeOp(PimpleCorrectorCondition()),
                    operation_name="pimple_corrector_loop",
                    operation_number=3,
                )

                with pimple_loop.loop(corrector_loop_op) as corrector_loop:
                    corrector_loop.step(ops["compute_HbyA"])
                    corrector_loop.step(ops["compute_phiHbyA"])
                    corrector_loop.step(ops["adjust_phi"])

                    # Non-orthogonal corrector loop
                    non_orth_loop_op = Operation(
                        func=IterativeOp(NonOrthogonalCondition()),
                        operation_name="non_orthogonal_loop",
                        operation_number=4,
                    )

                    with corrector_loop.loop(non_orth_loop_op) as non_orth_loop:
                        non_orth_loop.step(ops["solve_pressure"])
                        non_orth_loop.step(ops["update_flux"])

                    # After pressure correction
                    corrector_loop.step(ops["correct_velocity"])

                # After PIMPLE loop
                pimple_loop.step(ops["turbulence_correction"])

            # After time step
            time_loop.step(ops["write_output"])

        # Execute the operations
        main_loop.operations.run(ctx)

    def run(self) -> None:
        """Run the complete simulation."""
        # Initialize context
        ctx = self.create_context()

        # Initialize fields
        ops = self.operations()
        ops["create_fields"].run(ctx)
        ops["setup_models"].run(ctx)

        Info("Starting time loop")

        # Run main loop
        self.main_loop(ctx)

        Info("End")
