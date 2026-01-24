# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
SimpleSolver - FastAPI-style syntax with execution graph support.

The init module handles all initialization steps using FastAPI-style decorators.
"""

from typing import Any

import pybFoam as pyf
from pybFoam import Info
from pydantic import BaseModel

from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    StepBuilder,
    DAGResolver,
)
from foamadapter.framework.solver import Solver

# Import the init instance from simple_solver_init
from .create_fields import init


class TimeLoop:
    """Helper class for managing the main time loop operations."""

    def __call__(self, ctx: Context) -> bool:
        """Check if the time loop should continue running."""
        runTime = ctx.runTime
        return bool(runTime.run())


@Solver
class SimpleSolver(BaseModel):
    """SimpleSolver - FastAPI-style with execution graph support."""

    model_config = {"arbitrary_types_allowed": True}

    # Configuration
    argv: list[str] = []

    # Model storage for execution_graph() access
    _algorithm: Any = None
    _cfl_condition: Any = None
    _optional_models: list[Any] = []

    def initialize(self) -> Context:
        """
        Run init from the init module.

        The init instance is defined in simple_solver_init.py with
        FastAPI-style @init.step decorators.
        """
        init.argv = self.argv
        ctx = init.run()

        # Store model references for execution_graph()
        self._algorithm = ctx.models.get("algorithm")
        self._cfl_condition = ctx.models.get("cfl_condition")

        # Store optional models (NEW)
        optional_models = ctx.models.get("optional_models", [])
        self._optional_models = [
            m for m in optional_models if hasattr(m, "operations") and m.enabled
        ]

        if self._optional_models:
            Info(f"Loaded {len(self._optional_models)} optional model(s) into solver")

        return ctx


# ============================================================================
# EXECUTION GRAPH (like IncompressibleFluid)
# ============================================================================


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
    _ = domain_name

    # Get solver's own operations
    funcs = decorated_member_functions(self)
    solver_ops = OperationCollection()
    for func in funcs:
        op = Operation.create_SeqOp(func)
        solver_ops.add(op)

    # Get algorithm operations
    if self._algorithm is None:
        raise RuntimeError("Algorithm not found - call initialize() first")

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

    # Collect optional model operations (NEW)
    model_ops = OperationCollection()
    if hasattr(self, "_optional_models") and self._optional_models:
        for model in self._optional_models:
            if hasattr(model, "operations"):
                m_ops = model.operations()
                for op in m_ops:
                    model_ops.add(op)

    return main_loop, model_ops


# Add execution_graph as a method
SimpleSolver.execution_graph = execution_graph


# ============================================================================
# OPERATIONS
# ============================================================================


@Solver.operation
def set_time_step(self, ctx: Context) -> None:
    """Adjust time step based on CFL condition."""
    cfl_condition = ctx.models.get("cfl_condition")
    if cfl_condition:
        cfl_condition(ctx)


@Solver.operation
def increment_time(self, ctx: Context) -> None:
    """Print current simulation time and increment."""
    runTime = ctx.runTime
    Info(f"Time = {runTime.timeName()}")
    runTime.increment()


@Solver.operation(depends_on=["continuity"])
def turbulence_correction(self, ctx: Context) -> FieldUpdates:
    """Correct turbulence after pressure-velocity coupling."""
    laminarTransport = ctx.models.get("laminarTransport")
    turbulence = ctx.models.get("turbulence")

    if laminarTransport and hasattr(laminarTransport, "correct"):
        laminarTransport.correct()
    if turbulence and hasattr(turbulence, "correct"):
        turbulence.correct()

    return FieldUpdates({})


@Solver.operation(depends_on=["turbulence_correction"])
def write_output(self, ctx: Context) -> None:
    """Write fields to disk."""
    runTime = ctx.runTime
    runTime.write(True)
    runTime.printExecutionTime()


# Attach operations as methods
SimpleSolver.set_time_step = set_time_step
SimpleSolver.increment_time = increment_time
SimpleSolver.turbulence_correction = turbulence_correction
SimpleSolver.write_output = write_output


# ============================================================================
# MAIN LOOP (like IncompressibleFluid)
# ============================================================================


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


# Attach main_loop as a method
SimpleSolver.main_loop = main_loop


def run(argv: list[str] | None = None) -> None:
    """Run the complete simulation."""
    solver = SimpleSolver(argv=argv or [])
    ctx = solver.initialize()

    Info("Starting time loop")

    # Run main loop
    solver.main_loop(ctx)

    Info("End")
