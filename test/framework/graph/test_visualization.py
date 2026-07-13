# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the graph visualization helpers (DAG building and pyvis rendering)."""

from pathlib import Path

from neofoam.framework.graph import dependency_dag, digraph_to_pyvis_html
from neofoam.framework.graph.visualization import (
    _build_dag,
    _build_global_dag,
    _compute_nodes_order,
    _compute_steps_order,
)
from neofoam.framework.operations import (
    Operation,
    OperationCollection,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata, OperationNumber


def _meta(name, depends_on=None, operation_number=None, shape="box", color="lightblue"):
    return OperationMetadata(
        op_name=name,
        depends_on=depends_on or [],
        operation_number=operation_number,
        shape=shape,
        color=color,
    )


def _op(name, depends_on=None):
    return Operation(
        func=SequentialOp(lambda ctx: None),
        metadata=OperationMetadata(op_name=name, depends_on=depends_on or []),
    )


def test_build_dag_nodes_and_edges():
    nodes = [_meta("A"), _meta("B", depends_on=["A"], shape="circle", color="red")]

    graph = _build_dag(nodes)

    assert set(graph.nodes) == {"A", "B"}
    assert graph.has_edge("A", "B")
    assert graph.nodes["B"]["shape"] == "circle"
    assert graph.nodes["B"]["color"] == "red"
    assert graph.nodes["B"]["operation_number"] is None


def test_build_global_dag_composes_domains():
    domain1 = [_meta("A"), _meta("B", depends_on=["A"])]
    domain2 = [_meta("C"), _meta("D", depends_on=["C"])]

    graph = _build_global_dag({"d1": domain1, "d2": domain2})

    assert set(graph.nodes) == {"A", "B", "C", "D"}
    assert graph.has_edge("A", "B")
    assert graph.has_edge("C", "D")


def test_dependency_dag_public_entrypoint():
    """Public ``dependency_dag`` returns the same graph as ``_build_global_dag``."""
    domain1 = [_meta("A"), _meta("B", depends_on=["A"])]

    public = dependency_dag({"d": domain1})
    private = _build_global_dag({"d": domain1})

    assert set(public.nodes) == set(private.nodes)
    assert set(public.edges) == set(private.edges)


def test_compute_nodes_order_with_operation_number():
    nodes = [
        _meta("late", operation_number=OperationNumber(3)),
        _meta("unnumbered"),
        _meta("early", operation_number=OperationNumber(1)),
    ]

    order = _compute_nodes_order(nodes)

    assert order.index("early") < order.index("late")
    assert order.index("late") < order.index("unnumbered")


def test_compute_nodes_order_dependency_respected():
    nodes = [_meta("B", depends_on=["A"]), _meta("A")]

    order = _compute_nodes_order(nodes)

    assert order == ["A", "B"]


def test_compute_steps_order_returns_sorted_operations():
    op_a = _op("A")
    op_b = _op("B", depends_on=["A"])
    op_c = _op("C", depends_on=["B"])
    collection = OperationCollection(operations=[op_c, op_a, op_b])

    sorted_ops = _compute_steps_order(collection)

    assert [op.operation_name for op in sorted_ops.ops] == ["A", "B", "C"]
    assert sorted_ops.ops[0] is op_a
    assert sorted_ops.ops[1] is op_b
    assert sorted_ops.ops[2] is op_c


def test_digraph_to_pyvis_html_writes_file(tmp_path: Path):
    node1 = _meta("node1", operation_number=OperationNumber("1.0.0"), color="red")
    node2 = _meta(
        "node2",
        depends_on=["node1"],
        operation_number=OperationNumber("1.0.1"),
        color="blue",
    )

    graph = _build_dag([node1, node2])
    out = tmp_path / "dag.html"
    digraph_to_pyvis_html(graph, html_path=str(out))

    assert out.exists()
    contents = out.read_text()
    assert "node1" in contents and "node2" in contents
    # The node1 -> node2 dependency edge is rendered into the pyvis graph.
    assert '"from": "node1"' in contents and '"to": "node2"' in contents
    # Node labels are rendered so both operations are identifiable in the graph.
    assert '"label": "node1"' in contents and '"label": "node2"' in contents
