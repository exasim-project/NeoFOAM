# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Graph construction utilities."""

from typing import Mapping, Sequence

import networkx as nx  # type: ignore[import-untyped]


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
