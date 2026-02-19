# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Validation utilities for dependency DAGs."""

from collections import Counter
from typing import Mapping, Sequence

import networkx as nx  # type: ignore[import-untyped]

from .models import GraphDiagnostic, GraphValidationReport
from .sorter import NetworkxTopologicalSorter


def build_dependency_digraph(
    dependencies_by_node: Mapping[str, Sequence[str]],
) -> nx.DiGraph:
    """Build a directed graph where edges are dependency -> dependant."""
    graph = nx.DiGraph()

    for node_name, dependencies in dependencies_by_node.items():
        graph.add_node(node_name)
        for dependency in dependencies:
            graph.add_edge(dependency, node_name)

    return graph


def validate_dependency_graph(
    node_names: Sequence[str], dependencies_by_node: Mapping[str, Sequence[str]]
) -> GraphValidationReport:
    """Validate a dependency graph and return structured diagnostics."""
    diagnostics: list[GraphDiagnostic] = []
    name_counts = Counter(node_names)

    for node_name, count in sorted(name_counts.items()):
        if count > 1:
            diagnostics.append(
                GraphDiagnostic(
                    code="duplicate_name",
                    node_name=node_name,
                    message=f"Duplicate InitStep name detected: '{node_name}'",
                )
            )

    known_names = set(name_counts)
    for node_name, dependencies in dependencies_by_node.items():
        for dependency in dependencies:
            if dependency not in known_names:
                diagnostics.append(
                    GraphDiagnostic(
                        code="missing_dependency",
                        node_name=node_name,
                        dependency=dependency,
                        message=(
                            f"InitStep '{node_name}' depends on '{dependency}', "
                            f"but '{dependency}' was not found"
                        ),
                    )
                )

    if diagnostics:
        return GraphValidationReport(diagnostics=tuple(diagnostics))

    graph = build_dependency_digraph(dependencies_by_node)

    try:
        NetworkxTopologicalSorter().sort(graph)
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        cycle_edges = nx.find_cycle(graph)
        cycle_names = tuple(edge[0] for edge in cycle_edges)
        diagnostics.append(
            GraphDiagnostic(
                code="cycle",
                cycle=cycle_names,
                message=f"Circular dependency: {' -> '.join(cycle_names)}",
            )
        )

    return GraphValidationReport(diagnostics=tuple(diagnostics))
