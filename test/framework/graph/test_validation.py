# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for dependency-graph validation (duplicates, missing deps, cycles)."""

from neofoam.framework.graph import (
    build_dependency_digraph,
    validate_dependency_graph,
)


def test_validate_dependency_graph_duplicate_name() -> None:
    report = validate_dependency_graph(["A", "A"], {"A": []})

    assert not report.is_valid
    assert len(report.diagnostics) == 1
    assert report.diagnostics[0].code == "duplicate_name"


def test_validate_dependency_graph_missing_dependency() -> None:
    report = validate_dependency_graph(["A"], {"A": ["missing"]})

    assert not report.is_valid
    assert len(report.diagnostics) == 1
    assert report.diagnostics[0].code == "missing_dependency"


def test_validate_dependency_graph_cycle() -> None:
    report = validate_dependency_graph(["A", "B"], {"A": ["B"], "B": ["A"]})

    assert not report.is_valid
    assert len(report.diagnostics) == 1
    assert report.diagnostics[0].code == "cycle"


def test_validate_dependency_graph_valid_returns_empty_report():
    report = validate_dependency_graph(
        ["A", "B", "C"], {"A": [], "B": ["A"], "C": ["B"]}
    )

    assert report.is_valid
    assert report.diagnostics == ()


def test_validate_dependency_graph_self_dependency():
    report = validate_dependency_graph(["A"], {"A": ["A"]})

    assert not report.is_valid
    assert any(d.code == "cycle" for d in report.diagnostics)


def test_build_dependency_digraph_edge_direction():
    graph = build_dependency_digraph({"B": ["A"]})

    assert graph.has_edge("A", "B")
    assert not graph.has_edge("B", "A")
