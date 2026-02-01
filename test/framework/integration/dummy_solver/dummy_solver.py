# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
DummySolver - Test solver with FastAPI-style syntax.

Mimics SimpleSolver structure for testing new API.
"""

from typing import Annotated


from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.initialization import Depends, StagedInit
from foamadapter.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    StepBuilder,
    DAGResolver,
)
from foamadapter.framework.solver_factory import Solver

# Import create_init from dummy_init
from .dummy_init import create_init


class AlgorithmLoop:
    """Helper class for managing the algorithm inner loop."""

    def __init__(self):
        self._iteration = 0
        self._max_iterations = 3  # Limit for testing

    def __call__(self, ctx: Context) -> bool:
        """Check if algorithm iteration should continue."""
        self._iteration += 1
        if self._iteration > self._max_iterations:
            self._iteration = 0  # Reset for next time step
            return False

        algorithm = ctx.models.get("algorithm")
        if algorithm and hasattr(algorithm, "solve"):
            return algorithm.solve()
        return True  # Continue for testing


# Create Solver instance for decorating operations
dummy_solver = Solver("DummySolver")


@dummy_solver.initializer
def initialize(init: Annotated[StagedInit, Depends(create_init)]) -> Context:
    """Initialize using create_init factory with dependency injection."""
    init.argv = dummy_solver.argv
    return init.run()


@dummy_solver.execution_graph_step
def execution_graph(
    domain_name: str | None = None,
) -> tuple[StepBuilder, OperationCollection]:
    """
    Build solver structure and collect model operations.

    Returns:
        Tuple of (StepBuilder with solver structure, OperationCollection with model ops)
    """
    _ = domain_name

    # Build solver structure

    ops = dummy_solver.operations
    builder = StepBuilder()

    # Outer time loop - limit to 1 iteration for testing
    time_iteration = {"count": 0}

    def time_condition(ctx: Context) -> bool:
        time_iteration["count"] += 1
        return time_iteration["count"] <= 1  # Only 1 time step for testing

    time_loop = Operation(
        func=IterativeOp(time_condition),
        operation_name="time_loop",
        operation_number=None,
    )

    with builder.loop(time_loop) as time_builder:
        # Algorithm inner loop
        algo_loop = Operation(
            func=IterativeOp(AlgorithmLoop()),
            operation_name="inner_loop",
            operation_number=None,
        )

        with time_builder.loop(algo_loop) as inner_builder:
            inner_builder.step(ops["solver_step1"])
            inner_builder.step(ops["solver_step2"])
            inner_builder.step(ops["solver_step3"])

    # Collect model operations
    model_ops = OperationCollection()
    for model in dummy_solver.optional_models:
        model_ops.add(model.operations)

    return builder, model_ops


def run() -> Context:
    """
    Run the solver.

    Returns:
        Final context after solving
    """
    # Initialize
    ctx = dummy_solver.initialize()

    # Build and resolve execution graph
    builder, model_ops = dummy_solver.execution_graph()
    resolver = DAGResolver()
    resolved = resolver.resolve(builder, model_ops)

    # Run one time step (simplified)
    resolved.operations.run(ctx)

    return ctx


@dummy_solver.operation(operation_number="1.0")
def solver_step1(self, field1: float, field2: float) -> FieldUpdates:
    """
    Primary solver step.

    Args:
        self: Solver instance
        ctx: Execution context

    Returns:
        FieldUpdates with updated field1
    """
    # Update field1 based on field2
    f1_new = field1 + field2 * 1e-6 * 0.01  # dt = 0.01

    return FieldUpdates({"field1": f1_new})


@dummy_solver.operation(operation_number="2.0")
def solver_step2(self, field2: float, field3: float) -> FieldUpdates:
    """
    Secondary solver step.

    Args:
        self: Solver instance
        ctx: Execution context

    Returns:
        FieldUpdates with updated field2
    """
    # Update field2 based on field3
    f2_new = field2 - field3 * 1000.0

    return FieldUpdates({"field2": f2_new})


@dummy_solver.operation(operation_number="3.0")
def solver_step3(self, field1: float) -> FieldUpdates:
    """
    Correction solver step.

    Args:
        self: Solver instance
        ctx: Execution context

    Returns:
        FieldUpdates with corrected field1
    """
    # Simple correction
    f1_corrected = field1 * 0.99  # Small correction

    return FieldUpdates({"field1": f1_corrected})
