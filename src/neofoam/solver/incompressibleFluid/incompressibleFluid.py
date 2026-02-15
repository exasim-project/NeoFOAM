# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
SimpleSolver - FastAPI-style syntax with execution graph support.

The init module handles all initialization steps using FastAPI-style decorators.
"""

from typing import Annotated, Optional

from pybFoam import Info

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import Depends, StagedInit
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    StepBuilder,
)
from neofoam.framework.solver_factory import Solver

# Import create_init from create_fields
from .create_fields import create_init


class TimeLoop:
    """Helper class for managing the main time loop operations."""

    def __call__(self, ctx: Context) -> bool:
        """Check if the time loop should continue running."""
        runTime = ctx.runTime
        return bool(runTime.run())


# Create Solver instance for decorating operations
incompressibleFluid = Solver("incompressibleFluid")


@incompressibleFluid.initializer
def initialize(init: Annotated[StagedInit, Depends(create_init)]) -> Context:
    """Initialize using create_init factory with dependency injection."""
    init.argv = incompressibleFluid.argv
    ctx = init.run()
    incompressibleFluid.core_models = init.core_models
    incompressibleFluid.optional_models = init.optional_models
    return ctx


@incompressibleFluid.execution_graph_step
def execution_graph(
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, OperationCollection]:
    """
    Build solver structure and collect model operations.

    Returns:
        Tuple of (StepBuilder with solver structure, OperationCollection with model ops)
    """
    _ = domain_name

    ops = incompressibleFluid.operations
    builder = StepBuilder()

    # Get the algorithm model directly from core_models (it's the first item)
    # This is the detected algorithm model (pimple/simple/piso)
    algorithm_model = incompressibleFluid.core_models[0]
    algo_ops = algorithm_model.operations_for(algorithm_model)

    # Time loop structure
    time_loop_op = Operation(
        func=IterativeOp(TimeLoop()),
        operation_name="time_loop",
        operation_number=None,
    )

    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(ops["set_time_step"])
        time_builder.step(ops["increment_time"])

        # Algorithm inner loop
        with time_builder.loop(algo_ops["inner_loop"]) as inner_builder:
            inner_builder.step(algo_ops["momentum"])
            inner_builder.step(algo_ops["continuity"])
            inner_builder.step(ops["turbulence_correction"])

        time_builder.step(ops["write_output"])

    # Collect optional model operations
    model_ops = OperationCollection()
    for model in incompressibleFluid.optional_models:
        model_ops.add(model.operations)

    return builder, model_ops


def run(argv: Optional[list[str]] = None) -> Context:
    """
    Run the complete simulation.

    Args:
        argv: Command-line arguments

    Returns:
        Final context after solving
    """
    # Set argv
    incompressibleFluid.argv = argv or []

    # Initialize
    ctx = incompressibleFluid.initialize()

    Info("Starting time loop")

    # Build and resolve execution graph
    builder, model_ops = incompressibleFluid.execution_graph()
    resolver = DAGResolver()
    resolved = resolver.resolve(builder, model_ops)

    # Execute the resolved operations
    resolved.operations.run(ctx)

    Info("End")

    return ctx


# ============================================================================
# OPERATIONS
# ============================================================================


@incompressibleFluid.operation()
def set_time_step(self, ctx: Context) -> None:
    """Adjust time step based on CFL condition."""
    cfl_condition = ctx.models.get("cfl_condition")
    if cfl_condition:
        cfl_condition(ctx)


@incompressibleFluid.operation()
def increment_time(self, ctx: Context) -> None:
    """Print current simulation time and increment."""
    Info(f"Time = {ctx.runTime.timeName()}")
    ctx.runTime.increment()


@incompressibleFluid.operation(depends_on=["continuity"])
def turbulence_correction(self, ctx: Context) -> FieldUpdates:
    """Correct turbulence after pressure-velocity coupling."""
    laminarTransport = ctx.models.get("laminarTransport")
    turbulence = ctx.models.get("turbulence")

    if laminarTransport and hasattr(laminarTransport, "correct"):
        laminarTransport.correct()
    if turbulence and hasattr(turbulence, "correct"):
        turbulence.correct()

    return FieldUpdates({})


@incompressibleFluid.operation(depends_on=["turbulence_correction"])
def write_output(self, ctx: Context) -> None:
    """Write fields to disk."""
    ctx.runTime.write(True)
    ctx.runTime.printExecutionTime()
