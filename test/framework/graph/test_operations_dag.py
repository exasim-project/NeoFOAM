# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from neofoam.framework.graph import (
    build_dag,
    build_global_dag,
    compute_nodes_order,
    compute_steps_order,
)
from neofoam.framework.operations import Operation, OperationCollection
from neofoam.framework.types import OperationMetadata, OperationNumber


def _meta(name, depends_on=None, operation_number=None, shape="box", color="lightblue"):
    return OperationMetadata(
        op_name=name,
        depends_on=depends_on or [],
        operation_number=operation_number,
        shape=shape,
        color=color,
    )


def test_build_dag_nodes_and_edges():
    nodes = [_meta("A"), _meta("B", depends_on=["A"], shape="circle", color="red")]

    graph = build_dag(nodes)

    assert set(graph.nodes) == {"A", "B"}
    assert graph.has_edge("A", "B")
    assert graph.nodes["B"]["shape"] == "circle"
    assert graph.nodes["B"]["color"] == "red"
    assert graph.nodes["B"]["operation_number"] is None


def test_build_global_dag_composes_domains():
    domain1 = [_meta("A"), _meta("B", depends_on=["A"])]
    domain2 = [_meta("C"), _meta("D", depends_on=["C"])]

    graph = build_global_dag({"d1": domain1, "d2": domain2})

    assert set(graph.nodes) == {"A", "B", "C", "D"}
    assert graph.has_edge("A", "B")
    assert graph.has_edge("C", "D")


def test_compute_nodes_order_with_operation_number():
    nodes = [
        _meta("late", operation_number=OperationNumber(3)),
        _meta("unnumbered"),
        _meta("early", operation_number=OperationNumber(1)),
    ]

    order = compute_nodes_order(nodes)

    assert order.index("early") < order.index("late")
    assert order.index("late") < order.index("unnumbered")


def test_compute_nodes_order_dependency_respected():
    nodes = [_meta("B", depends_on=["A"]), _meta("A")]

    order = compute_nodes_order(nodes)

    assert order == ["A", "B"]


def test_compute_steps_order_returns_sorted_operations():
    op_a = Operation(func=lambda ctx: None, operation_name="A")
    op_b = Operation(func=lambda ctx: None, operation_name="B", depends_on=["A"])
    op_c = Operation(func=lambda ctx: None, operation_name="C", depends_on=["B"])
    collection = OperationCollection(operations=[op_c, op_a, op_b])

    sorted_ops = compute_steps_order(collection)

    assert [op.operation_name for op in sorted_ops.ops] == ["A", "B", "C"]
    assert sorted_ops.ops[0] is op_a
    assert sorted_ops.ops[1] is op_b
    assert sorted_ops.ops[2] is op_c
