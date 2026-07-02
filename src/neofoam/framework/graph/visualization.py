# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Visualization helpers for graph structures.

Public entry points render a :class:`networkx.DiGraph` for inspection, hand-off
to other tools, or interactive viewing:

* :func:`format_digraph` — a numbered text report (execution order).
* :func:`digraph_to_dot` — Graphviz ``DOT`` (edges: dependency → node).
* :func:`digraph_to_pyvis_html` — an interactive pyvis HTML page.
* :func:`write_digraph` — pick the format from the path suffix and write it.

These read an optional node-attribute vocabulary — ``category`` (str),
``write`` (bool), ``depends_on`` (sequence of names), plus the pyvis renderer's
``shape``/``color`` — so any producer can annotate its graph without the
renderers knowing the producer's domain. :func:`dependency_dag` composes
per-domain operation DAGs into a single graph for rendering or diagnostics.
"""

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import networkx as nx  # type: ignore[import-untyped]
from pyvis.network import Network  # type: ignore[import-untyped]

from neofoam.framework.operations import Operation, OperationCollection, Operations
from neofoam.framework.types import OperationMetadata

from .sorter import NetworkxTopologicalSorter


def _build_dag(nodes: list[OperationMetadata]) -> nx.DiGraph:
    """Build a DAG from a list of OperationMetadata objects."""
    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(
            node.name,
            meta=node,
            shape=node.shape,
            color=node.color,
            operation_number=node.operation_number,
        )
        for dependency in node.dependencies:
            graph.add_edge(dependency, node.name)
    return graph


def _build_global_dag(
    domains: Mapping[str, list[OperationMetadata]],
) -> nx.DiGraph:
    """Compose per-domain DAGs into a single global DAG."""
    graph = nx.DiGraph()
    for _, nodes in domains.items():
        sub_graph = _build_dag(nodes)
        graph = nx.compose(graph, sub_graph)
    return graph


def dependency_dag(domains: Mapping[str, list[OperationMetadata]]) -> nx.DiGraph:
    """Public entry point: build the global dependency DAG for ``domains``.

    Thin wrapper over :func:`_build_global_dag` so callers (e.g.
    :meth:`Simulation.dependency_graph`) bind to a stable public name.
    """
    return _build_global_dag(domains)


def operations_dag(
    operations: Operations, *, include_nesting: bool = True
) -> nx.DiGraph:
    """Build the operation DAG for an :class:`Operations` tree.

    Nodes are the operations, with loop/conditional bodies recursed into via
    ``sub_operations``. Two kinds of edge are drawn:

    * **dependency** edges, from each operation's declared ``depends_on``
      (dependency → operation);
    * **containment** edges, from a loop/conditional operation to each of its
      direct sub-operations (parent → child), when ``include_nesting`` is set.

    Without the containment edges most steps in a sequentially-built loop would
    render as disconnected nodes, since ordering there comes from the loop
    structure rather than explicit ``depends_on``. Suitable for rendering with
    :func:`digraph_to_dot` / :func:`write_digraph`.
    """
    graph = nx.DiGraph()

    def walk(op: Operation, parent: Optional[str]) -> None:
        meta = op.metadata
        name = meta.name
        graph.add_node(
            name,
            meta=meta,
            shape=meta.shape,
            color=meta.color,
            operation_number=meta.operation_number,
        )
        for dependency in meta.dependencies:
            graph.add_edge(dependency, name)
        if include_nesting and parent is not None:
            graph.add_edge(parent, name)
        for sub in op.sub_operations:
            walk(sub, name)

    for op in operations:
        walk(op, None)
    return graph


def operation_order(operations: Operations) -> list[str]:
    """Return operation names in execution order (pre-order walk of the tree).

    An :class:`Operations` tree runs top to bottom, descending into each
    loop/conditional's ``sub_operations`` in turn — so a depth-first pre-order
    walk yields the exact sequence in which the steps first execute. Use this as
    the ``order`` for :func:`digraph_to_dot` / :func:`format_digraph` so the node
    numbering matches the real run order rather than a re-derived sort.
    """
    order: list[str] = []

    def walk(op: Operation) -> None:
        name = op.metadata.name
        if name is not None:
            order.append(name)
        for sub in op.sub_operations:
            walk(sub)

    for op in operations:
        walk(op)
    return order


def _compute_nodes_order(nodes: list[OperationMetadata]) -> list[str]:
    """Deterministic topological order over the node metadata list."""
    graph = _build_dag(nodes)
    sorter = NetworkxTopologicalSorter(
        key=lambda node_name: (
            graph.nodes[node_name].get("operation_number") is None,
            graph.nodes[node_name].get("operation_number"),
            node_name,
        )
    )
    return sorter.sort(graph)


def _compute_steps_order(op_col: OperationCollection) -> Operations:
    """Order ``op_col``'s operations to match :func:`_compute_nodes_order`."""
    nodes = [op.metadata for op in op_col.ops]
    sorted_names = _compute_nodes_order(nodes)

    sorted_ops = Operations()
    name_to_op = {op.operation_name: op for op in op_col.ops}
    for node_name in sorted_names:
        sorted_ops.add(name_to_op[node_name])
    return sorted_ops


def digraph_to_pyvis_html(graph: nx.DiGraph, html_path: str = "dag.html") -> None:
    """Render a directed graph to a pyvis HTML file."""
    net = Network(directed=True, notebook=False)
    for node, attrs in graph.nodes(data=True):
        shape = attrs.get("shape", "ellipse")
        net.add_node(
            node, label=str(node), shape=shape, color=attrs.get("color", "lightblue")
        )
    for source, target in graph.edges:
        net.add_edge(source, target)

    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed",
                "nodeSpacing": 180,
                "levelSeparation": 150,
            }
        },
        "physics": {"enabled": True},
        "edges": {
            "arrows": {"to": {"enabled": True}},
            "smooth": False,
            "font": {"size": 14, "align": "middle"},
        },
    }

    net.set_options(json.dumps(options))
    net.write_html(html_path)


def _dot_escape(text: str) -> str:
    """Escape a string for use inside a DOT double-quoted id/label."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def digraph_to_dot(
    graph: nx.DiGraph,
    *,
    order: Optional[Sequence[str]] = None,
    name: str = "dag",
    rankdir: str = "LR",
) -> str:
    """Render ``graph`` as Graphviz ``DOT`` (edges: dependency → node).

    Every node and edge is drawn, so dependency keys that were never declared as
    nodes (external prerequisites, added by their edges) stay visible. For a node
    listed in ``order`` the label carries its 1-based execution index and, when
    present, its ``category`` attribute; other nodes render with just their name.
    """
    positions = {node_name: idx for idx, node_name in enumerate(order or [], start=1)}

    lines = [f"digraph {name} {{", f"  rankdir={rankdir};", "  node [shape=box];"]
    for node_name, attrs in graph.nodes(data=True):
        esc_name = _dot_escape(str(node_name))
        if node_name in positions:
            # ``\n`` is a DOT line-break directive — assemble it from
            # already-escaped parts so it is NOT doubled by the escaper.
            label = f"{positions[node_name]}: {esc_name}"
            category = attrs.get("category")
            if category:
                label += "\\n[" + _dot_escape(str(category)) + "]"
        else:
            label = esc_name
        lines.append(f'  "{esc_name}" [label="{label}"];')
    for source, target in graph.edges:
        lines.append(f'  "{_dot_escape(str(source))}" -> "{_dot_escape(str(target))}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def format_digraph(
    graph: nx.DiGraph,
    *,
    order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> str:
    """Render ``graph`` as a numbered text report (one node per line).

    Nodes are numbered in ``order`` (default: the graph's node order); nodes not
    in ``order`` — e.g. external dependency keys — are omitted from the listing.
    Each line shows the node's name, its ``category`` attribute, its dependencies
    (the ``depends_on`` attribute if set, else the graph predecessors), and a
    ``*write`` flag when the ``write`` attribute is truthy.
    """
    node_names = list(order) if order is not None else list(graph.nodes)
    name_width = max((len(n) for n in node_names), default=4)
    cat_width = max(
        (len(str(graph.nodes[n].get("category", ""))) for n in node_names),
        default=8,
    )

    lines = []
    if title:
        lines.append(f"# {title}\n")
    for idx, node_name in enumerate(node_names, start=1):
        attrs = graph.nodes[node_name]
        category = str(attrs.get("category", ""))
        declared = attrs.get("depends_on")
        deps_list = (
            list(declared)
            if declared is not None
            else sorted(graph.predecessors(node_name))
        )
        deps = ", ".join(deps_list) if deps_list else "-"
        flags = " *write" if attrs.get("write") else ""
        lines.append(
            f"{idx:>4}. {node_name:<{name_width}}  "
            f"[{category:<{cat_width}}]  deps: {deps}{flags}"
        )
    return "\n".join(lines) + "\n"


def write_digraph(
    graph: nx.DiGraph,
    path: Union[str, Path],
    *,
    order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> Path:
    """Write ``graph`` to ``path``; the format is chosen from the suffix.

    ``.dot`` / ``.gv`` → Graphviz DOT (:func:`digraph_to_dot`); ``.html`` /
    ``.htm`` → interactive pyvis HTML (:func:`digraph_to_pyvis_html`); anything
    else → the numbered text report (:func:`format_digraph`). Returns the path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix in (".dot", ".gv"):
        out.write_text(digraph_to_dot(graph, order=order))
    elif suffix in (".html", ".htm"):
        digraph_to_pyvis_html(graph, html_path=str(out))
    else:
        out.write_text(format_digraph(graph, order=order, title=title))
    return out
