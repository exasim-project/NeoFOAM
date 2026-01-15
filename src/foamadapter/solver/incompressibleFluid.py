# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from typing import Any, Literal

import pybFoam as pyf
from pybFoam import (
    Info,
)
from pydantic import BaseModel

from foamadapter.algorithms.pressure_velocity import (
    PressureVelocityAlgorithm,
)
from foamadapter.framework.context import (
    Context,
    FieldUpdates,
)
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    StepBuilder,
    DAGResolver,
)
from foamadapter.framework.solver import Solver

from foamadapter.models.stability_criteria import CFLCondition
from foamadapter.models.transport_model import TransportModel
from foamadapter.models.turbulence import TurbulenceModel


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
    _algorithm_config: Any | None = (
        None  # Algorithm instance (PimpleMethod/etc) for BUILD stage
    )
    _fvSolution: Any | None = (
        None  # Keep fvSolution alive to prevent C++ object destruction
    )
    _cfl_number: CFLCondition | None = None

    # === Optional Physics Models ===
    models: list[Any] = []
    _has_boussinesq: bool = False  # Set during resolve_dependencies

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
        """Set pressure reference for algorithm (algorithm already created in LOAD)."""
        p = context["fields.p"]
        mesh = context["mesh"]

        # Algorithm already exists from LOAD stage, just set pressure reference
        # Pass p_rgh if available (Boussinesq mode)
        p_rgh = context.get("fields.p_rgh", None)
        assert self._pressure_velocity is not None
        self._pressure_velocity.set_pressure_reference(p, mesh, self._fvSolution, p_rgh)

        return self._pressure_velocity

    @Solver.load
    def load_control_dict(self) -> dict[str, Any]:
        """LOAD: Initialize CFL condition and algorithm instance."""
        # CFLCondition reads its own config from controlDict
        self._cfl_number = CFLCondition()

        # Create algorithm instance (fields will be set up in BUILD stage)
        self._fvSolution = pyf.dictionary.read("system/fvSolution")
        self._pressure_velocity = PressureVelocityAlgorithm.from_fvSolution(
            self._fvSolution
        )

        # Return algorithm for automatic registration in ConfigContext
        return {"algorithm": self._pressure_velocity}

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

        # Algorithm was created in LOAD stage, now set up its fields
        # Note: pRefCell/pRefValue set later in _create_algorithm
        assert self._pressure_velocity is not None
        initializers.extend(self._pressure_velocity.setup())

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
    ) -> FieldUpdates:
        """
        Correct turbulence model after pressure-velocity coupling.

        For SIMPLE: always correct turbulence every iteration
        For PIMPLE: only correct when turbCorr() returns true
        """
        # For SIMPLE, always correct. For PIMPLE, check control object.
        should_correct = True

        if should_correct:
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

    def operations(
        self, domain_name: str | None = None
    ) -> tuple[StepBuilder, OperationCollection]:
        """
        Build solver structure and collect model operations.

        Returns a tuple of:
        1. StepBuilder with solver/algorithm structure (time loop, inner loop)
        2. OperationCollection with model operations to be inserted

        The DAG resolver will merge these, respecting dependencies.
        """
        _ = domain_name  # Part of SolverInterface, unused in this implementation

        # Build solver + algorithm structure

        # Get solver's own operations
        funcs = decorated_member_functions(self)
        solver_ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            solver_ops.add(op)

        assert self._pressure_velocity is not None
        algo_ops = self._pressure_velocity.operations()

        # Build the structural StepBuilder
        main_loop = StepBuilder()

        # Time loop structure
        time_loop_op = Operation(
            func=IterativeOp(TimeLoop()),
            operation_name="time_loop",
        )

        with main_loop.loop(time_loop_op) as time_loop:
            time_loop.step(solver_ops["set_time_step"])
            time_loop.step(solver_ops["increment_time"])

            with time_loop.loop(algo_ops["inner_loop"]) as iloop:
                iloop.step(algo_ops["momentum"])
                iloop.step(algo_ops["continuity"])
                iloop.step(solver_ops["turbulence_correction"])

            time_loop.step(solver_ops["write_output"])

        # Collect optional model operations (for extending the solver)
        model_ops = OperationCollection()

        # Add optional model operations (buoyancy, etc.)
        for model in self.models:
            m_ops = model.operations()
            for op in m_ops:
                model_ops.add(op)

        return main_loop, model_ops

    def main_loop(self, ctx: Context) -> None:
        """
        Main simulation loop using DAG resolver.

        The DAG resolver merges solver structure with model operations,
        respecting all dependencies.
        """
        if self._pressure_velocity is None:
            raise RuntimeError(
                "Algorithm not initialized. Call initialize() or run() first."
            )

        # Get solver structure and model operations
        step_builder, model_ops = self.operations()

        # Resolve operation ordering with DAG resolver
        resolver = DAGResolver()
        resolved_builder = resolver.resolve(step_builder, model_ops)

        # Execute the resolved operations
        resolved_builder.operations.run(ctx)

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
