# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from typing import Any, Literal

import pybFoam as pyf
from pybFoam import (
    Info,
)
from pydantic import BaseModel

from foamadapter.algorithms.pressure_velocity import (
    PressureVelocityAlgorithmConfig,
)
from foamadapter.framework.context import (
    Context,
    FieldUpdates,
    Model as ModelAnnotation,
)
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.initialization import ConfigContext
from foamadapter.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    StepBuilder,
)
from foamadapter.framework.solver import Solver

from foamadapter.models.stability_criteria import CFLCondition
from foamadapter.models.transport_model import TransportModel
from foamadapter.models.turbulence import TurbulenceModel
from foamadapter.models.incompressible_fluid_model import IncompressibleFluidModel


class TimeLoop:
    """Helper class for managing the main time loop operations."""

    def __call__(self, ctx: Context) -> bool:
        """Check if the time loop should continue running."""
        runTime = ctx.runTime
        return bool(runTime.run())


@Solver
class IncompressibleFluid(BaseModel):
    """
    Incompressible fluid solver with modular physics.

    Core components (always present, type-configurable):
    - pressure_velocity: Algorithm for pressure-velocity coupling
    - transport: Transport properties model
    - turbulence: Turbulence model

    Optional models:
    - Buoyancy, radiation, species transport, etc.
    """

    model_config = {"arbitrary_types_allowed": True}

    # === Configuration (minimal - components read from files) ===
    name: Literal["IncompressibleFluid"] = "IncompressibleFluid"
    argv: list[str] = []

    # === Core Components (private, populated during initialization) ===
    _pressure_velocity: Any | None = None
    _transport: Any | None = None
    _turbulence: Any | None = None
    _algorithm_config: dict[str, Any] | None = (
        None  # Algorithm configuration for BUILD stage
    )
    _fvSolution: Any | None = (
        None  # Keep fvSolution alive to prevent C++ object destruction
    )
    _cfl_number: CFLCondition | None = None

    # === Optional Physics Models ===
    models: list[IncompressibleFluidModel] = []

    def __repr__(self) -> str:
        """Custom repr to avoid OpenFOAM SIGFPE issues in pytest."""
        return "IncompressibleFluid()"

    def get_models(self) -> list[Any]:
        """
        Return all models owned by this solver.

        Returns models that have lifecycle methods: algorithm and optional models.
        """
        models = []
        # Add algorithm (has lifecycle methods)
        if self._pressure_velocity is not None:
            models.append(self._pressure_velocity)

        # Add optional physics models
        models.extend(self.models)
        return models

    def _create_algorithm(self, context: dict[str, Any]) -> Any:
        """Create algorithm instance from context dict (reads fvSolution)."""
        p = context["fields.p"]
        mesh = context["mesh"]

        # Read solver configuration
        # IMPORTANT: Keep fvSolution in scope to prevent C++ object destruction
        self._fvSolution = pyf.dictionary.read("system/fvSolution")

        # Algorithm detects its type and sets pressure reference
        self._pressure_velocity = PressureVelocityAlgorithmConfig.from_fvSolution(
            self._fvSolution
        )
        self._pressure_velocity.set_pressure_reference(p, mesh, self._fvSolution)

        return self._pressure_velocity

    @Solver.load
    def load_control_dict(self) -> None:
        """LOAD: Initialize CFL condition (reads maxDeltaT from controlDict)."""
        # CFLCondition reads its own config from controlDict
        self._cfl_number = CFLCondition()

    @Solver.resolve_dependencies
    def configure_solver(self, config: ConfigContext) -> None:
        """RESOLVE_DEPENDENCIES: Minimal - components self-configure from files."""
        # Components will read their own configuration from OpenFOAM files
        # during BUILD stage. Nothing to configure here.
        pass

    def _create_transport(self, ctx: dict[str, Any]) -> Any:
        """Create transport model instance (reads config from constant/transportProperties)."""
        return TransportModel.from_type("singlePhase").create_instance(
            ctx["fields.U"], ctx["fields.phi"]
        )

    def _create_turbulence(self, ctx: dict[str, Any]) -> Any:
        """Create turbulence model instance (reads config from constant/momentumTransport)."""
        return TurbulenceModel.from_type("openfoam_rts").create_instance(
            ctx["fields.U"], ctx["fields.phi"], ctx["fields.laminarTransport"]
        )

    @Solver.build
    def setup_runtime(self, mesh: Any) -> list[Any]:
        """BUILD: Initialize runtime structures and fields."""
        from foamadapter.framework.initialization.helpers import field, lazy
        from foamadapter.foam.initialization import create_time_mesh

        # Build initialization list (runtime + mesh)
        initializers = create_time_mesh(self.argv)

        # Algorithm detects type from fvSolution and sets up its fields
        # Note: pRefCell/pRefValue set later in _create_algorithm
        fvSolution = pyf.dictionary.read("system/fvSolution")
        algorithm = PressureVelocityAlgorithmConfig.from_fvSolution(fvSolution)
        initializers.extend(algorithm.setup())

        # Transport and turbulence models (OpenFOAM reads config from files)
        initializers.extend(
            [
                field(
                    "laminarTransport",
                    depends_on=["fields.U", "fields.phi"],
                    create=self._create_transport,
                ),
                field(
                    "turbulence",
                    depends_on=["fields.U", "fields.phi", "fields.laminarTransport"],
                    create=self._create_turbulence,
                ),
            ]
        )

        # Algorithm instance created after its fields exist
        initializers.append(
            lazy(
                "algorithm",
                depends_on=["fields.p", "mesh"],
                create=self._create_algorithm,
            )
        )

        return initializers

    @Solver.operation
    def set_time_step(self, ctx: Context) -> None:
        assert self._cfl_number is not None
        self._cfl_number(ctx)

    @Solver.operation
    def increment_time(self, ctx: Context) -> None:
        """Print current simulation time."""
        runTime = ctx.runTime
        Info(f"Time = {runTime.timeName()}")
        runTime.increment()

    @Solver.operation(depends_on=["continuity"])
    def turbulence_correction(
        self,
        laminarTransport: Any,
        turbulence: Any,
        pimple_control: ModelAnnotation[Any],
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

    @Solver.operation(depends_on=["turbulence_correction"])
    def write_output(self, ctx: Context) -> None:
        """Write fields to disk."""
        runTime = ctx.runTime
        runTime.write(True)
        runTime.printExecutionTime()

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        """
        Collect all operations from solver and models.

        Operations are collected from multiple sources and merged into a single
        collection. The execution order is determined by operation numbers and
        dependencies.

        Returns:
            OperationCollection with all operations from:
            - Solver's own @Solver.operation decorated methods
            - Pressure-velocity algorithm
            - Optional physics models (buoyancy, radiation, etc.)
        """
        _ = domain_name  # Part of SolverInterface, unused in this implementation

        # Collect solver operations
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)

        # Add algorithm operations if initialized
        # Note: algorithm may not be initialized yet during early operations() calls
        if self._pressure_velocity is not None:
            algo_ops = self._pressure_velocity.operations()
            ops.add(algo_ops)

        # Add optional model operations
        for model in self.models:
            if hasattr(model, "operations"):
                model_ops = model.operations()
                ops.add(model_ops)

        return ops

    def main_loop(self, ctx: Context) -> None:
        """
        Main simulation loop - algorithm agnostic!

        The algorithm provides momentum and continuity operations that encapsulate
        the specific pressure-velocity coupling strategy.
        """
        if self._pressure_velocity is None:
            raise RuntimeError(
                "Algorithm not initialized. Call initialize() or run() first."
            )

        ops = self.operations()
        algo_ops = self._pressure_velocity.operations()

        # Build the main execution graph
        main_loop = StepBuilder()

        # Time loop
        time_loop_op = Operation(
            func=IterativeOp(TimeLoop()),
            operation_name="time_loop",
        )

        with main_loop.loop(time_loop_op) as time_loop:
            time_loop.step(ops["set_time_step"])
            time_loop.step(ops["increment_time"])

            # Algorithm provides momentum and continuity operations
            with time_loop.loop(algo_ops["inner_loop"]) as iloop:
                iloop.step(algo_ops["momentum"])
                iloop.step(algo_ops["continuity"])

                # Solver handles turbulence correction
                iloop.step(ops["turbulence_correction"])

            time_loop.step(ops["write_output"])

        # Execute the operations
        main_loop.operations.run(ctx)

    def run(self) -> None:
        """Run the complete simulation using 3-stage initialization."""
        from foamadapter.framework.initialization import SolverInitializer

        # 3-stage initialization (includes model setup via algorithm.setup())
        initializer = SolverInitializer(self)
        ctx = initializer.initialize(mesh=None)

        Info("Starting time loop")

        # Run main loop
        self.main_loop(ctx)

        Info("End")
