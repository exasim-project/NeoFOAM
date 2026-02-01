# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for cycle detection and dependency validation in DAG resolver."""

import pytest
from foamadapter.framework.operations import (
    Operation,
    OperationCollection,
    StepBuilder,
    SequentialOp,
    IterativeOp,
    DAGResolver,
    CyclicDependencyError,
    MissingDependencyError,
)
from .dag_test_helpers import noop, time_loop_condition


@pytest.mark.parametrize(
    "ops_config,test_id",
    [
        # Simple cycle: A -> B -> A
        ([("A", ["B"]), ("B", ["A"])], "simple_cycle"),
        # Self cycle: A -> A
        ([("A", ["A"])], "self_cycle"),
        # Long cycle: A -> B -> C -> D -> B
        ([("A", []), ("B", ["A", "D"]), ("C", ["B"]), ("D", ["C"])], "long_cycle"),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_cycle_detection(ops_config, test_id):
    """Test detection of cyclic dependencies."""
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(time_loop_condition),
        operation_name="time_loop",
    )

    with builder.loop(time_loop) as tloop:
        for name, deps in ops_config:
            tloop.step(
                Operation(
                    func=SequentialOp(noop),
                    operation_name=name,
                    depends_on=deps,
                )
            )

    resolver = DAGResolver()
    model_ops = OperationCollection()

    with pytest.raises(CyclicDependencyError):
        resolver.resolve(builder, model_ops)


def test_missing_dependency():
    """Test detection of missing dependency."""
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(time_loop_condition),
        operation_name="time_loop",
    )

    with builder.loop(time_loop) as tloop:
        tloop.step(
            Operation(
                func=SequentialOp(noop),
                operation_name="A",
                depends_on=["NonExistent"],
            )
        )

    resolver = DAGResolver()
    model_ops = OperationCollection()

    with pytest.raises(MissingDependencyError):
        resolver.resolve(builder, model_ops)


def test_cross_scope_dependency_invalid():
    """Test that operations are placed in scope with their dependencies."""
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(time_loop_condition),
        operation_name="time_loop",
    )

    with builder.loop(time_loop) as tloop:
        tloop.step(
            Operation(
                func=SequentialOp(noop),
                operation_name="outer_op",
            )
        )

    # Add model op that depends on operation in different scope
    # The resolver should place it in the same scope as its dependency
    model_op = Operation(
        func=SequentialOp(noop),
        operation_name="root_op",
        depends_on=["outer_op"],  # This is in time_loop scope
    )
    model_op.loop_context = "root"  # Requested root, but will be placed in time_loop
    model_ops = OperationCollection([model_op])

    resolver = DAGResolver()

    # Should successfully resolve by placing root_op in time_loop scope
    result = resolver.resolve(builder, model_ops)

    # Verify root_op was placed in time_loop scope with outer_op
    time_loop_result = result.operations[0]
    op_names = {op.operation_name for op in time_loop_result.sub_operations}
    assert "outer_op" in op_names
    assert "root_op" in op_names
