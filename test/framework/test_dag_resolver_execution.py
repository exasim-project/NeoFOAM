# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests that verify operations execute in correct order with correct results."""

from foamadapter.framework.operations import (
    Operation,
    OperationCollection,
    StepBuilder,
    SequentialOp,
    IterativeOp,
    DAGResolver,
)
from foamadapter.framework.context import Context
from .dag_test_helpers import (
    set_value,
    double_value,
    add_ten,
    multiply_by_three,
    momentum,
    continuity,
    turbulence_correction,
    update_buoyancy,
    time_loop_condition,
    inner_loop_condition,
)


def test_simple_execution_order():
    """Test that operations execute in correct dependency order."""
    builder = StepBuilder()

    # Build operations: set_value (42) -> double_value (84) -> add_ten (94)
    builder.step(
        Operation(
            func=SequentialOp(add_ten),
            operation_name="add_ten",
            depends_on=["double_value"],
        )
    )
    builder.step(
        Operation(
            func=SequentialOp(set_value),
            operation_name="set_value",
        )
    )
    builder.step(
        Operation(
            func=SequentialOp(double_value),
            operation_name="double_value",
            depends_on=["set_value"],
        )
    )

    resolver = DAGResolver()
    result = resolver.resolve(builder, OperationCollection())

    # Execute operations
    ctx = Context(fields={}, models={})
    for op in result.operations:
        op.func(ctx)

    # Verify correct result: 42 * 2 + 10 = 94
    assert ctx.fields["value"] == 94


def test_parallel_operations_execution():
    """Test that independent operations can execute in any order."""
    builder = StepBuilder()

    # Two independent operations
    builder.step(
        Operation(
            func=SequentialOp(lambda ctx: ctx.fields.__setitem__("a", 10)),
            operation_name="set_a",
        )
    )
    builder.step(
        Operation(
            func=SequentialOp(lambda ctx: ctx.fields.__setitem__("b", 20)),
            operation_name="set_b",
        )
    )
    # Dependent operation
    builder.step(
        Operation(
            func=SequentialOp(
                lambda ctx: ctx.fields.__setitem__(
                    "sum", ctx.fields["a"] + ctx.fields["b"]
                )
            ),
            operation_name="compute_sum",
            depends_on=["set_a", "set_b"],
        )
    )

    resolver = DAGResolver()
    result = resolver.resolve(builder, OperationCollection())

    # Execute operations
    ctx = Context(fields={}, models={})
    for op in result.operations:
        op.func(ctx)

    # Verify correct result
    assert ctx.fields["a"] == 10
    assert ctx.fields["b"] == 20
    assert ctx.fields["sum"] == 30


def test_model_operation_execution():
    """Test that model operations execute correctly when inserted."""
    builder = StepBuilder()

    # Builder operations
    builder.step(
        Operation(
            func=SequentialOp(set_value),
            operation_name="set_value",
        )
    )

    # Model operation that depends on set_value
    model_op = Operation(
        func=SequentialOp(multiply_by_three),
        operation_name="multiply_by_three",
        depends_on=["set_value"],
    )
    model_op.loop_context = "root"

    resolver = DAGResolver()
    result = resolver.resolve(builder, OperationCollection([model_op]))

    # Execute operations
    ctx = Context(fields={}, models={})
    for op in result.operations:
        op.func(ctx)

    # Verify correct result: 42 * 3 = 126
    assert ctx.fields["value"] == 126


def test_nested_loop_execution():
    """Test execution in nested loops with dependencies."""
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(time_loop_condition),
        operation_name="time_loop",
    )

    with builder.loop(time_loop) as tloop:
        tloop.step(
            Operation(
                func=SequentialOp(lambda ctx: ctx.fields.__setitem__("time", 0)),
                operation_name="init_time",
            )
        )

        inner_loop = Operation(
            func=IterativeOp(inner_loop_condition),
            operation_name="inner_loop",
        )
        with tloop.loop(inner_loop) as iloop:
            iloop.step(
                Operation(
                    func=SequentialOp(momentum),
                    operation_name="momentum",
                )
            )
            iloop.step(
                Operation(
                    func=SequentialOp(continuity),
                    operation_name="continuity",
                    depends_on=["momentum"],
                )
            )

    # Add model operation
    model_op = Operation(
        func=SequentialOp(turbulence_correction),
        operation_name="turbulence_correction",
        depends_on=["continuity"],
    )
    model_op.loop_context = "inner_loop"

    resolver = DAGResolver()
    result = resolver.resolve(builder, OperationCollection([model_op]))

    # Execute time_loop operations (just the first iteration for testing)
    ctx = Context(fields={}, models={})
    ctx.fields["time"] = 0
    ctx.fields["end_time"] = 1
    ctx.fields["max_iter"] = 1
    ctx.fields["iteration"] = 0

    # Get inner loop
    time_loop_ops = result.operations[0].sub_operations
    init_time_op = time_loop_ops[0]
    inner_loop_op = time_loop_ops[1]

    # Execute init_time
    init_time_op.func(ctx)
    assert ctx.fields["time"] == 0

    # Execute inner loop operations
    for op in inner_loop_op.sub_operations:
        op.func(ctx)

    # Verify execution order was correct:
    # 1. momentum sets velocity to 1.0
    # 2. continuity sets pressure to velocity * 2.0 = 2.0
    # 3. turbulence_correction sets turbulence to pressure * 0.5 = 1.0
    assert ctx.fields["velocity"] == 1.0
    assert ctx.fields["pressure"] == 2.0
    assert ctx.fields["turbulence"] == 1.0


def test_complex_dependency_execution():
    """Test complex scenario with multiple dependencies executing correctly."""
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(time_loop_condition),
        operation_name="time_loop",
    )

    with builder.loop(time_loop) as tloop:
        inner_loop = Operation(
            func=IterativeOp(inner_loop_condition),
            operation_name="inner_loop",
        )
        with tloop.loop(inner_loop) as iloop:
            iloop.step(
                Operation(
                    func=SequentialOp(momentum),
                    operation_name="momentum",
                )
            )
            iloop.step(
                Operation(
                    func=SequentialOp(continuity),
                    operation_name="continuity",
                    depends_on=["momentum"],
                )
            )

    # Add model operations with dependencies
    buoyancy_op = Operation(
        func=SequentialOp(update_buoyancy),
        operation_name="update_buoyancy",
        depends_on=[],  # Independent
    )
    buoyancy_op.loop_context = "inner_loop"

    turbulence_op = Operation(
        func=SequentialOp(turbulence_correction),
        operation_name="turbulence_correction",
        depends_on=["continuity"],  # Depends on continuity
    )
    turbulence_op.loop_context = "inner_loop"

    resolver = DAGResolver()
    result = resolver.resolve(
        builder, OperationCollection([buoyancy_op, turbulence_op])
    )

    # Execute
    ctx = Context(fields={}, models={})
    ctx.fields["time"] = 0
    ctx.fields["end_time"] = 1
    ctx.fields["max_iter"] = 1
    ctx.fields["iteration"] = 0

    # Get and execute inner loop operations
    inner_loop_op = result.operations[0].sub_operations[0]

    for op in inner_loop_op.sub_operations:
        op.func(ctx)

    # Verify all operations executed with correct dependencies
    assert ctx.fields["buoyancy"] == 9.81  # update_buoyancy (independent)
    assert ctx.fields["velocity"] == 1.0  # momentum (independent)
    assert ctx.fields["pressure"] == 2.0  # continuity (depends on momentum)
    assert (
        ctx.fields["turbulence"] == 1.0
    )  # turbulence_correction (depends on continuity)
