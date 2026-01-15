# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for topological sorting in DAG resolver."""

import pytest
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
    time_loop_condition,
)


@pytest.mark.parametrize(
    "ops_config,expected_order,test_id",
    [
        # Simple chain: A -> B -> C
        ([("C", ["B"]), ("A", []), ("B", ["A"])], ["A", "B", "C"], "simple_chain"),
        # Parallel operations (no dependencies)
        ([("A", []), ("B", []), ("C", [])], ["A", "B", "C"], "parallel"),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_topological_sorting(ops_config, expected_order, test_id):
    """Test topological sort with various dependency patterns."""
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
    result = resolver.resolve(builder, OperationCollection())

    time_loop_ops = [op.operation_name for op in result.operations[0].sub_operations]
    for i, expected_name in enumerate(expected_order):
        assert time_loop_ops[i] == expected_name


def test_diamond_dependency():
    """Test topological sort with diamond dependency: A -> B,C -> D."""
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(time_loop_condition),
        operation_name="time_loop",
    )

    with builder.loop(time_loop) as tloop:
        tloop.step(
            Operation(
                func=SequentialOp(noop),
                operation_name="D",
                depends_on=["B", "C"],
            )
        )
        tloop.step(
            Operation(
                func=SequentialOp(noop),
                operation_name="B",
                depends_on=["A"],
            )
        )
        tloop.step(
            Operation(
                func=SequentialOp(noop),
                operation_name="A",
            )
        )
        tloop.step(
            Operation(
                func=SequentialOp(noop),
                operation_name="C",
                depends_on=["A"],
            )
        )

    resolver = DAGResolver()
    model_ops = OperationCollection()
    result = resolver.resolve(builder, model_ops)

    time_loop_ops = [op.operation_name for op in result.operations[0].sub_operations]

    # A must be first
    assert time_loop_ops[0] == "A"
    # D must be last
    assert time_loop_ops[3] == "D"
    # B and C can be in either order at positions 1,2
    assert time_loop_ops[1] in {"B", "C"}
    assert time_loop_ops[2] in {"B", "C"}
