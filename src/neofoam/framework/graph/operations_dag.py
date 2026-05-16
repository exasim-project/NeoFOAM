# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""DAG helpers for operation metadata graphs."""

import networkx as nx  # type: ignore[import-untyped]

from neofoam.framework.types import OperationMetadata


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
