# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from pathlib import Path

import networkx as nx

from neofoam.framework.graph import (
    dependency_dag,
    digraph_to_dot,
    digraph_to_pyvis_html,
    format_digraph,
    operation_order,
    operations_dag,
    write_digraph,
)
from neofoam.framework.graph.visualization import (
    _build_dag,
    _build_global_dag,
    _compute_nodes_order,
    _compute_steps_order,
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
    from neofoam.framework.operations import SequentialOp

    def _op(name, depends_on=None):
        return Operation(
            func=SequentialOp(lambda ctx: None),
            metadata=OperationMetadata(op_name=name, depends_on=depends_on or []),
        )

    op_a = _op("A")
    op_b = _op("B", depends_on=["A"])
    op_c = _op("C", depends_on=["B"])
    collection = OperationCollection(operations=[op_c, op_a, op_b])

    sorted_ops = _compute_steps_order(collection)

    assert [op.operation_name for op in sorted_ops.ops] == ["A", "B", "C"]
    assert sorted_ops.ops[0] is op_a
    assert sorted_ops.ops[1] is op_b
    assert sorted_ops.ops[2] is op_c


def _annotated_diamond() -> nx.DiGraph:
    """A diamond with an external dependency ``root`` that is not a node.

    Steps ``a``/``b`` depend on ``root`` (never declared), ``c`` depends on both.
    ``b`` is flagged for writing.
    """
    graph = nx.DiGraph()
    graph.add_node("a", category="fields", write=False, depends_on=["root"])
    graph.add_node("b", category="models", write=True, depends_on=["root"])
    graph.add_node("c", category="fields", write=False, depends_on=["a", "b"])
    for source, target in [("root", "a"), ("root", "b"), ("a", "c"), ("b", "c")]:
        graph.add_edge(source, target)
    return graph


def test_digraph_to_dot_has_nodes_and_edges():
    dot = digraph_to_dot(_annotated_diamond(), order=["a", "b", "c"], name="init")

    assert dot.startswith("digraph init {")
    assert dot.rstrip().endswith("}")
    # edges are drawn dependency -> node, including the external ``root``
    assert '"root" -> "a";' in dot
    assert '"a" -> "c";' in dot
    # an ordered node's label carries its execution index and category, with a
    # real DOT line-break (single backslash-n), not an escaped literal ``\\n``
    assert 'label="1: a\\n[fields]"' in dot
    assert "\\\\n" not in dot


def test_format_digraph_numbers_in_order_and_omits_externals():
    text = format_digraph(
        _annotated_diamond(),
        order=["a", "b", "c"],
        title="my dag",
    )
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]

    assert "# my dag" in text
    assert lines[0].startswith("   1. a")
    # dependencies come from the declared ``depends_on`` attribute
    assert "deps: a, b" in text
    # the write flag is surfaced for step ``b``
    assert "*write" in text
    # only the three declared steps are numbered; the external ``root`` (which
    # still appears as a dependency) is not listed as a step of its own
    assert len(lines) == 3
    assert not any(ln.split(".", 1)[1].strip().startswith("root") for ln in lines)


def test_write_digraph_picks_format_from_suffix(tmp_path: Path):
    graph = _annotated_diamond()
    order = ["a", "b", "c"]

    txt = write_digraph(graph, tmp_path / "dag.txt", order=order, title="t")
    dot = write_digraph(graph, tmp_path / "dag.dot", order=order)
    html = write_digraph(graph, tmp_path / "dag.html", order=order)

    assert txt.read_text().startswith("# t")
    assert dot.read_text().startswith("digraph dag {")
    assert "<html" in html.read_text().lower()


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


def _op(name, depends_on=None, sub_operations=None):
    from neofoam.framework.operations import SequentialOp

    return Operation(
        func=SequentialOp(lambda ctx: None),
        metadata=OperationMetadata(op_name=name, depends_on=depends_on or []),
        sub_operations=sub_operations or [],
    )


def test_operations_dag_captures_nesting_and_dependencies():
    momentum = _op("momentum")
    continuity = _op("continuity", depends_on=["momentum"])
    inner = _op("inner_loop", sub_operations=[momentum, continuity])
    time_loop = _op("time_loop", sub_operations=[inner])
    operations = OperationCollection(operations=[time_loop])

    graph = operations_dag(operations)

    assert set(graph.nodes) == {"time_loop", "inner_loop", "momentum", "continuity"}
    # containment edges: parent -> each direct child
    assert graph.has_edge("time_loop", "inner_loop")
    assert graph.has_edge("inner_loop", "momentum")
    assert graph.has_edge("inner_loop", "continuity")
    # dependency edge from ``depends_on``
    assert graph.has_edge("momentum", "continuity")


def test_operation_order_is_preorder_walk():
    momentum = _op("momentum")
    continuity = _op("continuity", depends_on=["momentum"])
    inner = _op("inner_loop", sub_operations=[momentum, continuity])
    time_loop = _op("time_loop", sub_operations=[inner, _op("write_output")])
    operations = OperationCollection(operations=[time_loop])

    # depth-first pre-order: a parent precedes its children, siblings in order
    assert operation_order(operations) == [
        "time_loop",
        "inner_loop",
        "momentum",
        "continuity",
        "write_output",
    ]


def test_operations_dag_without_nesting_drops_containment_edges():
    child = _op("child")
    parent = _op("parent", sub_operations=[child])
    operations = OperationCollection(operations=[parent])

    graph = operations_dag(operations, include_nesting=False)

    assert set(graph.nodes) == {"parent", "child"}
    # no containment edge when nesting is disabled
    assert not graph.has_edge("parent", "child")
