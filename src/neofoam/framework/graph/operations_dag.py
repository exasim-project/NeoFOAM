# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""DAG helpers for operation metadata graphs."""

import networkx as nx  # type: ignore[import-untyped]

from neofoam.framework.operations import OperationCollection, Operations
from neofoam.framework.types import OperationMetadata

from .sorter import NetworkxTopologicalSorter


def build_dag(nodes: list[OperationMetadata]) -> nx.DiGraph:
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


def build_global_dag(domains: dict[str, list[OperationMetadata]]) -> nx.DiGraph:
    """Build a global DAG from multiple domain models."""
    graph = nx.DiGraph()
    for _, nodes in domains.items():
        sub_graph = build_dag(nodes)
        graph = nx.compose(graph, sub_graph)
    return graph


def compute_nodes_order(nodes: list[OperationMetadata]) -> list[str]:
    """Compute a valid deterministic topological order of nodes."""
    graph = build_dag(nodes)
    sorter = NetworkxTopologicalSorter(
        key=lambda node_name: (
            graph.nodes[node_name].get("operation_number") is None,
            graph.nodes[node_name].get("operation_number"),
            node_name,
        )
    )
    return sorter.solve(graph)


def compute_steps_order(op_col: OperationCollection) -> Operations:
    """Compute a valid topological order of operations."""
    nodes = [op.operation_metadata() for op in op_col.ops]
    sorted_names = compute_nodes_order(nodes)

    sorted_ops = Operations()
    name_to_op = {op.name: op for op in op_col.ops}
    for node_name in sorted_names:
        sorted_ops.add(name_to_op[node_name])
    return sorted_ops
