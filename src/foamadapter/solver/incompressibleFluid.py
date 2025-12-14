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

from foamadapter.algorithms.pressure_velocity import PimpleAlgorithm
from foamadapter.framework.context import (
    Context,
    FieldUpdates,
    Model as ModelAnnotation,
)
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


@Solver
class IncompressibleFluid(BaseModel):
    """
    Incompressible fluid solver supporting multiple pressure-velocity algorithms.
    """

    name: Literal["IncompressibleFluid"] = "IncompressibleFluid"
    argv: list[str] = []
    algorithm: Literal["SIMPLE", "PISO", "PIMPLE"] = "PIMPLE"
    pRefCell: int | None = None
    pRefValue: float | None = None
    maxDeltaT: float = 1e5

    def _create_algorithm(self) -> PimpleAlgorithm:
        """Factory method to create algorithm instance."""
        if self.algorithm == "PIMPLE":
            return PimpleAlgorithm(pRefCell=self.pRefCell, pRefValue=self.pRefValue)
        else:
            raise ValueError(
                f"Algorithm '{self.algorithm}' not yet implemented. Only PIMPLE is currently supported."
            )

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
        """Setup algorithm control object."""
        # Remove old pimple from fields if it exists
        ctx.fields.pop("pimple", None)

        # Create algorithm control using algorithm's factory method
        algorithm = self._create_algorithm()
        control = algorithm.create_control(ctx.mesh)
        ctx.models["pimple_control"] = control

    @Solver.operation(operation_number=3, depends_on=["setup_models"])
    def print_time(self, ctx: Context) -> None:
        """Print current simulation time."""
        runTime = ctx.runTime
        Info(f"Time = {runTime.timeName()}")

    @Solver.operation(operation_number=4, depends_on=["continuity"])
    def turbulence_correction(
        self, laminarTransport, turbulence, pimple_control: ModelAnnotation
    ) -> FieldUpdates:
        """
        Correct turbulence model after pressure-velocity coupling.
        """
        if pimple_control.turbCorr():
            laminarTransport.correct()
            turbulence.correct()

        return FieldUpdates(
            {"laminarTransport": laminarTransport, "turbulence": turbulence}
        )

    @Solver.operation(operation_number=5, depends_on=["turbulence_correction"])
    def write_output(self, ctx: Context) -> None:
        """Write fields to disk."""
        runTime = ctx.runTime
        runTime.write(True)
        runTime.printExecutionTime()

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        """Collect all decorated operations including algorithm operations."""
        _ = domain_name  # Part of SolverInterface, unused in this implementation
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)

        # Add algorithm operations
        algorithm = self._create_algorithm()
        algo_ops = algorithm.operations()
        ops.add(algo_ops)

        return ops

    def main_loop(self, ctx: Context) -> None:
        """
        Main simulation loop - algorithm agnostic!

        The algorithm provides momentum and continuity operations that encapsulate
        the specific pressure-velocity coupling strategy.
        """
        ops = self.operations()
        algorithm = self._create_algorithm()

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

            # Algorithm provides momentum and continuity operations
            algo_ops = algorithm.operations()
            time_loop.step(algo_ops["momentum"])
            time_loop.step(algo_ops["continuity"])

            # Solver handles turbulence correction
            time_loop.step(ops["turbulence_correction"])

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
