# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the networkx topological sorter (ordering and tie-breaking)."""

from neofoam.framework.graph import (
    NetworkxTopologicalSorter,
    build_dependency_digraph,
)


def test_networkx_topological_sorter_orders_graph():
    graph = build_dependency_digraph({"B": ["A"], "C": ["B"], "A": []})

    order = NetworkxTopologicalSorter().sort(graph)
    assert order == ["A", "B", "C"]


def test_sorter_custom_key_breaks_ties():
    graph = build_dependency_digraph({"A": [], "B": [], "C": []})

    priority = {"A": 2, "B": 0, "C": 1}
    order = NetworkxTopologicalSorter(key=lambda node: priority[node]).sort(graph)
    assert order == ["B", "C", "A"]


def test_sorter_returns_list_type():
    graph = build_dependency_digraph({"A": [], "B": ["A"]})

    order = NetworkxTopologicalSorter().sort(graph)
    assert isinstance(order, list)
