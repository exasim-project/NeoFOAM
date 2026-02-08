# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
import networkx as nx  # type: ignore[import-untyped]

from foamadapter.framework.operations import OperationCollection, Operations

from .types import OperationMetadata


def build_dag(nodes: list[OperationMetadata]) -> nx.DiGraph:
    """
    Build a DAG from a list of OperationMetadata objects.
    """
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(
            node.name,
            meta=node,
            shape=node.shape,
            color=node.color,
            operation_number=node.operation_number,
        )
        for dep in node.dependencies:
            G.add_edge(dep, node.name)
    return G


def build_global_dag(domains: dict[str, list[OperationMetadata]]) -> nx.DiGraph:
    """
    Build a global DAG from multiple domain models, supporting interdomain dependencies.
    Each node is named as 'domain.step'.
    """
    G = nx.DiGraph()
    for domain_name, nodes in domains.items():
        sub_graph = build_dag(nodes)
        G = nx.compose(G, sub_graph)
    return G


def compute_nodes_order(nodes: list[OperationMetadata]) -> list[str]:
    """
    Compute a valid topological order of nodes in the DAG.
    """
    dag = build_dag(nodes)
    nodes_sorted = list(
        nx.lexicographical_topological_sort(
            dag, key=lambda n: dag.nodes[n]["operation_number"]
        )
    )
    return nodes_sorted


def compute_steps_order(op_col: OperationCollection) -> Operations:
    """
    Compute a valid topological order of steps in the DAG.
    """
    nodes = [op.operation_metadata() for op in op_col.ops]
    nodes_sorted = compute_nodes_order(nodes)
    ops_sorted = Operations()
    name_to_op = {op.name: op for op in op_col.ops}
    for node_name in nodes_sorted:
        ops_sorted.add(name_to_op[node_name])
    return ops_sorted
