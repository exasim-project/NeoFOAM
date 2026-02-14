# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from neofoam.framework.graph import (
    NetworkxTopologicalSorter,
    build_dependency_digraph,
    validate_dependency_graph,
)


def test_validate_dependency_graph_duplicate_name():
    report = validate_dependency_graph(["A", "A"], {"A": []})

    assert not report.is_valid
    assert len(report.diagnostics) == 1
    assert report.diagnostics[0].code == "duplicate_name"


def test_validate_dependency_graph_missing_dependency():
    report = validate_dependency_graph(["A"], {"A": ["missing"]})

    assert not report.is_valid
    assert len(report.diagnostics) == 1
    assert report.diagnostics[0].code == "missing_dependency"


def test_validate_dependency_graph_cycle():
    report = validate_dependency_graph(["A", "B"], {"A": ["B"], "B": ["A"]})

    assert not report.is_valid
    assert len(report.diagnostics) == 1
    assert report.diagnostics[0].code == "cycle"


def test_networkx_topological_sorter_orders_graph():
    graph = build_dependency_digraph({"B": ["A"], "C": ["B"], "A": []})

    order = NetworkxTopologicalSorter().sort(graph)
    assert order == ["A", "B", "C"]
