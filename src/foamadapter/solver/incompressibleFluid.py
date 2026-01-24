# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from typing import Any, Literal

import pybFoam as pyf
from pybFoam import (
    Info,
)
from pydantic import BaseModel

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
    DAGResolver,
)
from foamadapter.framework.solver import Solver
from foamadapter.framework.initialization.execution import execute_initialization
from foamadapter.framework.initialization.config_context import ConfigContext


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

    # Model storage for execution_graph() access
    _algorithm: Any = None
    _cfl_condition: Any = None
    _optional_models: list[Any] = []

    @property
    def core_models(self) -> list[Any]:
        """Core models that define solver structure."""
        return [m for m in [self._algorithm, self._cfl_condition] if m is not None]

    def initialize(self) -> Context:
        """Run 3-stage initialization with explicit stage calls."""
        from .incompressible_fluid_initializer import IncompressibleFluidInitializer

        initializer = IncompressibleFluidInitializer(self.argv)

        # === STAGE 1: LOAD ===
        config_items = initializer.load()

        # Validate
        errors = initializer.validate_load()
        for e in errors:
            if e.severity == "error":
                raise RuntimeError(f"Initialization error: {e.field}: {e.message}")
            else:
                Info(f"Warning: {e.field}: {e.message}")

        # === Build ConfigContext ===
        config = ConfigContext()
        for key, value in config_items.items():
            config.register(key, value)
        for model in initializer.data.optional_models:
            config.register(model.name, model)

        # === STAGE 2: RESOLVE ===
        initializer.resolve(config)

        # Validate
        warnings = initializer.validate_resolve(config)
        for w in warnings:
            Info(f"Warning: {w.field}: {w.message}")

        # === STAGE 3: BUILD ===
        lazy_inits = initializer.build()

        # === Execute ===
        ctx = execute_initialization(lazy_inits)

        # === Store model references for execution_graph() ===
        self._algorithm = initializer.data.algorithm
        self._cfl_condition = initializer.data.cfl_condition
        self._optional_models = [
            m
            for m in initializer.data.optional_models
            if hasattr(m, "execution_graph") or hasattr(m, "operations")
        ]

        return ctx

    def execution_graph(
        self, domain_name: str | None = None
    ) -> tuple[StepBuilder, OperationCollection]:
        """
        Build solver structure and collect model operations.

        Returns a tuple of:
        1. StepBuilder with solver/algorithm structure (time loop, inner loop)
        2. OperationCollection with model operations to be inserted

        The DAG resolver will merge these, respecting dependencies.
        """
        # Build solver + algorithm structure
        _ = domain_name

        # Get solver's own operations
        funcs = decorated_member_functions(self)
        solver_ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            solver_ops.add(op)

        if self._algorithm is None:
            raise RuntimeError("Algorithm not found")

        algo_ops = self._algorithm.execution_graph()

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
        for model in self._optional_models:
            if hasattr(model, "execution_graph"):
                m_ops = model.execution_graph()
                model_ops.add(m_ops)
            elif hasattr(model, "operations"):
                for op in model.operations():
                    model_ops.add(op)

        return main_loop, model_ops

    @Solver.operation
    def set_time_step(self, ctx: Context) -> None:
        cfl_condition = ctx.models.get("cfl_condition")
        if cfl_condition:
            cfl_condition(ctx)

    @Solver.operation
    def increment_time(self, ctx: Context) -> None:
        """Print current simulation time."""
        runTime = ctx.runTime
        Info(f"Time = {runTime.timeName()}")
        runTime.increment()

    @Solver.operation(depends_on=["continuity"])
    def turbulence_correction(
        self,
        laminarTransport: ModelAnnotation[Any],
        turbulence: ModelAnnotation[Any],
    ) -> FieldUpdates:
        """
        Correct turbulence model after pressure-velocity coupling.
        """

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

    def main_loop(self, ctx: Context) -> None:
        """
        Main simulation loop using DAG resolver.

        The DAG resolver merges solver structure with model operations,
        respecting all dependencies.
        """
        # Get solver structure and model operations
        solver_ops, model_ops = self.execution_graph()

        # Resolve operation ordering with DAG resolver
        resolver = DAGResolver()
        resolved_builder = resolver.resolve(solver_ops, model_ops)

        # Execute the resolved operations
        resolved_builder.operations.run(ctx)

    def run(self) -> None:
        """Run the complete simulation."""
        ctx = self.initialize()

        Info("Starting time loop")

        # Run main loop
        self.main_loop(ctx)

        Info("End")

    def __repr__(self) -> str:
        """Custom repr to avoid OpenFOAM SIGFPE issues in pytest."""
        return "IncompressibleFluid()"
