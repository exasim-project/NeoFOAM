# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
SimpleSolver - FastAPI-style syntax with execution graph support.

The init module handles all initialization steps using FastAPI-style decorators.
"""

from typing import Annotated, Any, Optional, Protocol

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import Depends, StagedInit
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    StepBuilder,
)
from neofoam.framework.solver import Solver
from neofoam.framework.types import OperationMetadata
from pybFoam import Info
from .create_fields import create_init


class CorrectableModel(Protocol):
    def correct(self) -> None: ...


class TimeLoop:
    """Helper class for managing the main time loop operations."""

    def __call__(self, ctx: Context) -> bool:
        """Check if the time loop should continue running."""
        return bool(ctx.runtime.run())


# Create Solver instance for decorating operations
incompressibleFluid = Solver("incompressibleFluid")


@incompressibleFluid.initializer
def initialize(self: Any, init: Annotated[StagedInit, Depends(create_init)]) -> Context:
    """Initialize using create_init factory with dependency injection."""
    ctx = init.run()
    return ctx


@incompressibleFluid.execution_graph_step
def execution_graph(
    self: Any,
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, Operations]:
    """
    Build solver structure and collect model operations.

    Returns:
        Tuple of (StepBuilder with solver structure, Operations with model ops)
    """
    _ = domain_name

    ops = self.operations
    builder = StepBuilder()

    # Get the algorithm model directly from core_models (it's the first item)
    # This is the detected algorithm model (pimple/simple/piso)
    algorithm_model = self.state.core_models[0]
    algo_ops = Operations(algorithm_model._build_operations_for(algorithm_model))

    # Time loop structure
    time_loop_op = Operation(
        func=IterativeOp(TimeLoop()),
        metadata=OperationMetadata(op_name="time_loop"),
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
    model_ops = Operations()
    for model in self.state.optional_models:
        model_ops.add(model.operations)

    return builder, model_ops


def run(
    argv: Optional[list[str]] = None,
    log_file: Optional[Any] = None,
) -> Context:
    """
    Run the complete simulation.

    Args:
        argv: Command-line arguments
        log_file: Optional path to redirect C++ stdout to a log file

    Returns:
        Final context after solving
    """
    import os
    import sys
    from pathlib import Path

    redirect = log_file is not None
    if redirect:
        log_path = Path(log_file)
        sys.stdout.flush()
        saved_fd = os.dup(1)
        log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log_fd, 1)
        os.close(log_fd)

    try:
        # Initialize
        solver = incompressibleFluid.instantiate(argv=argv or [])
        ctx = solver.initialize()

        Info("Starting time loop")

        # Build and resolve execution graph
        builder, model_ops = solver.execution_graph()
        resolver = DAGResolver()
        resolved = resolver.resolve(builder, model_ops)

        # Execute the resolved operations
        resolved.operations.run(ctx)

        Info("End")

        return ctx
    finally:
        if redirect:
            sys.stdout.flush()
            os.dup2(saved_fd, 1)
            os.close(saved_fd)


@incompressibleFluid.operation()
def set_time_step(
    self: Any,
    ctx: Context,
    pressure_velocity: Annotated[Optional[Any], "models"],
    cfl_condition: Annotated[Optional[Any], "models"],
) -> None:
    """Adjust time step based on CFL condition."""
    if (
        pressure_velocity is not None
        and getattr(pressure_velocity, "algorithm_type", "").upper() == "SIMPLE"
    ):
        return

    if cfl_condition:
        cfl_condition(ctx)


@incompressibleFluid.operation()
def increment_time(self: Any, ctx: Context) -> None:
    """Print current simulation time and increment."""
    Info(f"Time = {ctx.runtime.timeName()}")
    ctx.runtime.increment()


@incompressibleFluid.operation(depends_on=["continuity"])
def turbulence_correction(
    self: Any,
    laminarTransport: Annotated[Optional[CorrectableModel], "models"],
    turbulence: Annotated[Optional[CorrectableModel], "models"],
) -> FieldUpdates:
    """Correct turbulence after pressure-velocity coupling."""
    if laminarTransport:
        laminarTransport.correct()
    if turbulence:
        turbulence.correct()

    return FieldUpdates({})


@incompressibleFluid.operation(depends_on=["turbulence_correction"])
def write_output(self: Any, ctx: Context) -> None:
    """Write fields to disk."""
    ctx.runtime.write(True)
    ctx.runtime.printExecutionTime()
