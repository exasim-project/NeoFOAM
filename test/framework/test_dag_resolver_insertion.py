# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for operation insertion in DAG resolver."""

from foamadapter.framework.operations import (
    Operation,
    OperationCollection,
    StepBuilder,
    SequentialOp,
    IterativeOp,
    DAGResolver,
)
from dag_test_helpers import (
    noop,
    set_value,
    double_value,
    time_loop_condition,
)


def test_insert_at_scope_start():
    """Test inserting model operation at start of scope."""
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(time_loop_condition),
        operation_name="time_loop",
    )

    with builder.loop(time_loop) as tloop:
        tloop.step(
            Operation(
                func=SequentialOp(noop),
                operation_name="existing",
            )
        )

    model_op = Operation(
        func=SequentialOp(noop),
        operation_name="model_op",
        depends_on=[],
    )
    model_op.loop_context = "time_loop"
    model_ops = OperationCollection([model_op])

    resolver = DAGResolver()
    result = resolver.resolve(builder, model_ops)

    # Get operations in time_loop
    time_loop_ops = result.operations[0].sub_operations
    assert len(time_loop_ops) == 2
    # Both operations have no dependencies
    op_names = {op.operation_name for op in time_loop_ops}
    assert op_names == {"existing", "model_op"}


def test_insert_after_dependency():
    """Test inserting model operation after its dependency."""
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(time_loop_condition),
        operation_name="time_loop",
    )

    with builder.loop(time_loop) as tloop:
        tloop.step(
            Operation(
                func=SequentialOp(set_value),
                operation_name="set_value",
            )
        )

    model_op = Operation(
        func=SequentialOp(double_value),
        operation_name="double_value",
        depends_on=["set_value"],
    )
    model_op.loop_context = "time_loop"
    model_ops = OperationCollection([model_op])

    resolver = DAGResolver()
    result = resolver.resolve(builder, model_ops)

    # Get operations in time_loop as list
    time_loop_ops = [op.operation_name for op in result.operations[0].sub_operations]
    assert len(time_loop_ops) == 2

    # set_value should come before double_value
    assert time_loop_ops[0] == "set_value"
    assert time_loop_ops[1] == "double_value"


def test_insert_multiple_operations():
    """Test inserting multiple model operations."""
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(time_loop_condition),
        operation_name="time_loop",
    )

    with builder.loop(time_loop) as tloop:
        tloop.step(
            Operation(
                func=SequentialOp(set_value),
                operation_name="set_value",
            )
        )

    # Model operations with chain: op1 -> op2 -> op3
    op1 = Operation(
        func=SequentialOp(noop),
        operation_name="op1",
        depends_on=["set_value"],
    )
    op1.loop_context = "time_loop"

    op2 = Operation(
        func=SequentialOp(noop),
        operation_name="op2",
        depends_on=["op1"],
    )
    op2.loop_context = "time_loop"

    op3 = Operation(
        func=SequentialOp(noop),
        operation_name="op3",
        depends_on=["op2"],
    )
    op3.loop_context = "time_loop"

    model_ops = OperationCollection([op1, op2, op3])

    resolver = DAGResolver()
    result = resolver.resolve(builder, model_ops)

    # Get operations in time_loop as list
    time_loop_ops = [op.operation_name for op in result.operations[0].sub_operations]
    assert len(time_loop_ops) == 4

    # Check correct ordering
    assert time_loop_ops[0] == "set_value"
    assert time_loop_ops[1] == "op1"
    assert time_loop_ops[2] == "op2"
    assert time_loop_ops[3] == "op3"
