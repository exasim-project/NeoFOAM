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
from foamadapter.framework.initialization import ModelRegistry
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

    model_config = {"arbitrary_types_allowed": True}

    name: Literal["IncompressibleFluid"] = "IncompressibleFluid"
    argv: list[str] = []
    algorithm: Literal["SIMPLE", "PISO", "PIMPLE"] = "PIMPLE"
    pRefCell: int | None = None
    pRefValue: float | None = None
    maxDeltaT: float = 1e5

    # Lifecycle state tracking
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False

    # Model references (populated during initialization)
    transport_model: Any | None = None
    turbulence_model: Any | None = None
    algorithm_model: Any | None = None

    # Runtime objects (populated during SETUP)
    mesh: Any | None = None
    runTime: Any | None = None
    p: Any | None = None
    U: Any | None = None
    phi: Any | None = None

    def _create_algorithm(self) -> PimpleAlgorithm:
        """Factory method to create algorithm instance."""
        if self.algorithm == "PIMPLE":
            return PimpleAlgorithm(pRefCell=self.pRefCell, pRefValue=self.pRefValue)
        else:
            raise ValueError(
                f"Algorithm '{self.algorithm}' not yet implemented. Only PIMPLE is currently supported."
            )

    def get_models(self) -> list[Any]:
        """Return all models owned by this solver."""
        models = []
        if self.algorithm_model is not None:
            models.append(self.algorithm_model)
        # Note: transport_model and turbulence_model are OpenFOAM objects,
        # not Python models with lifecycle methods, so we don't include them
        return models

    @Solver.read_files
    def load_control_dict(self) -> None:
        """READ_FILES: Load solver control settings from configuration files."""
        # Read controlDict
        controlDict = pyf.dictionary.read("system/controlDict")
        try:
            self.maxDeltaT = controlDict.get[float]("maxDeltaT")
        except KeyError:
            pass  # Use default value

        # Read fvSolution for reference cell/value
        # Note: This requires mesh, so actual reading is deferred to SETUP
        # We just mark files as read here
        self.files_read = True

    @Solver.configure
    def configure_solver(self, registry: ModelRegistry) -> None:
        """CONFIGURE: Validate solver configuration and connect models."""
        # Create and register algorithm model
        self.algorithm_model = self._create_algorithm()
        registry.register("algorithm", self.algorithm_model)

        # Validate algorithm choice
        if self.algorithm not in ["SIMPLE", "PISO", "PIMPLE"]:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        self.configured = True

    @Solver.setup
    def setup_runtime(self, mesh: Any) -> None:
        """SETUP: Initialize runtime structures and fields."""
        # Create runtime and mesh
        argList = pyf.argList(self.argv)
        self.runTime = pyf.Time(argList)
        self.mesh = pyf.fvMesh(self.runTime)

        # Read fields
        self.p = volScalarField.read_field(self.mesh, "p")
        self.U = volVectorField.read_field(self.mesh, "U")
        self.phi = pyf.createPhi(self.U)

        # Create transport and turbulence models
        self.transport_model = singlePhaseTransportModel(self.U, self.phi)
        self.turbulence_model = incompressibleTurbulenceModel.New(
            self.U, self.phi, self.transport_model
        )

        # Read fvSolution and set reference cell
        fvSolution = pyf.dictionary.read("system/fvSolution")
        self.pRefCell, self.pRefValue = pyf.setRefCell(
            self.p, fvSolution.subDict("PIMPLE")
        )
        self.mesh.setFluxRequired(pyf.Word("p"))

        # Update algorithm with reference cell/value
        if self.algorithm_model is not None:
            self.algorithm_model.pRefCell = self.pRefCell
            self.algorithm_model.pRefValue = self.pRefValue

        self.setup_complete = True

    def create_context(self) -> Context:
        """Create the simulation context with mesh, runtime, and fields."""
        # After 3-stage initialization, mesh and runTime are already created
        if self.mesh is None or self.runTime is None:
            raise RuntimeError(
                "Solver not properly initialized. Call SolverInitializer.initialize() first."
            )

        # Reuse fields that were created during SETUP
        # Create pimple control
        pimple = pyf.pimpleControl(self.mesh)

        # Create context
        ctx = Context(
            fields={
                "p": self.p,
                "U": self.U,
                "phi": self.phi,
                "laminarTransport": self.transport_model,
                "turbulence": self.turbulence_model,
                "pimple": pimple,
            },
            models={},
            mesh=self.mesh,
            runTime=self.runTime,
        )

        return ctx

    @Solver.operation(operation_number=2)
    def setup_models(self, ctx: Context) -> None:
        """Setup algorithm control object."""
        # Remove old pimple from fields if it exists
        ctx.fields.pop("pimple", None)

        # Create algorithm control using algorithm's factory method
        if self.algorithm_model is None:
            raise RuntimeError(
                "Algorithm model not initialized. Call initialize() first."
            )
        control = self.algorithm_model.create_control(ctx.mesh)
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
        if self.algorithm_model is None:
            # Create algorithm lazily for tests/scenarios without full initialization
            self.algorithm_model = self._create_algorithm()
        algo_ops = self.algorithm_model.operations()
        ops.add(algo_ops)

        return ops

    def main_loop(self, ctx: Context) -> None:
        """
        Main simulation loop - algorithm agnostic!

        The algorithm provides momentum and continuity operations that encapsulate
        the specific pressure-velocity coupling strategy.
        """
        ops = self.operations()
        if self.algorithm_model is None:
            # Create algorithm lazily for tests/scenarios without full initialization
            self.algorithm_model = self._create_algorithm()

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
        """Run the complete simulation using 3-stage initialization."""
        from foamadapter.framework.initialization import SolverInitializer

        # 3-stage initialization
        initializer = SolverInitializer(self)
        initializer.initialize(mesh=None)

        # Create context with initialized fields
        ctx = self.create_context()

        # Setup models (algorithm control)
        ops = self.operations()
        ops["setup_models"].run(ctx)

        Info("Starting time loop")

        # Run main loop
        self.main_loop(ctx)

        Info("End")
