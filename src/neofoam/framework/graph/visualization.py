# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Visualization helpers for graph structures.

Public entry points are :func:`digraph_to_pyvis_html` (HTML rendering) and
:func:`dependency_dag` (compose per-domain operation DAGs into a single graph
for rendering or diagnostics).
"""

import json
from typing import Mapping

import networkx as nx  # type: ignore[import-untyped]
from pyvis.network import Network  # type: ignore[import-untyped]

from neofoam.framework.operations import OperationCollection, Operations
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
