# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the snakemake DOT → canvas graph rendering (UI-free)."""

from __future__ import annotations

import shutil

import pytest

from neofoam.tooling.workflow.dag import (
    dag_graph,
    layered_layout,
    parse_dot,
    snakemake_dot,
)

_DOT = """digraph snakemake_dag {
    graph[bgcolor=white, margin=0];
    node[shape=box, style=rounded, fontname=sans, fontsize=10, penwidth=2];
    edge[penwidth=2, color=grey];
\t0[label = "all", color = "0.33 0.6 0.85", style="rounded"];
\t1[label = "solve\\ncase: base", color = "0.00 0.6 0.85", style="rounded"];
\t2[label = "setup\\ncase: base", color = "0.66 0.6 0.85", style="rounded"];
\t1 -> 0
\t2 -> 1
}
"""


def test_parse_dot_extracts_nodes_and_edges():
    nodes, edges = parse_dot(_DOT)
    assert set(nodes) == {"0", "1", "2"}
    assert nodes["1"]["label"] == "solve\ncase: base"
    assert nodes["0"]["color"].startswith("#")
    assert edges == [("1", "0"), ("2", "1")]


def test_layered_layout_orders_dependents_below_dependencies():
    nodes, edges = parse_dot(_DOT)
    layout = layered_layout(nodes, edges)
    # Edges point dependency -> dependent, so `all` (node 0) sits lowest.
    assert layout["2"][1] < layout["1"][1] < layout["0"][1]


def test_snakemake_dot_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="unknown snakemake graph mode"):
        snakemake_dot(tmp_path, "jobgraph")


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="snakemake not installed")
def test_dag_graph_over_a_minimal_snakefile(tmp_path):
    (tmp_path / "Snakefile").write_text(
        'rule all:\n    input:\n        "done"\n\n'
        'rule work:\n    output:\n        "done"\n    shell:\n        "touch done"\n'
    )
    nodes, edges = dag_graph(tmp_path, "dag")
    labels = {n["data"]["label"] for n in nodes}
    assert {"all", "work"} <= labels
    assert len(edges) == 1
    node = nodes[0]
    assert set(node) >= {"id", "type", "position", "width", "height", "data", "style"}


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="snakemake not installed")
def test_dag_graph_surfaces_snakemake_errors(tmp_path):
    (tmp_path / "Snakefile").write_text("this is not a Snakefile\n")
    with pytest.raises(RuntimeError, match="snakemake --dag failed"):
        dag_graph(tmp_path, "dag")


@pytest.mark.skipif(
    shutil.which("snakemake") is None or shutil.which("dot") is None,
    reason="needs both snakemake and graphviz dot",
)
def test_graphviz_layout_places_dependents_below_dependencies(tmp_path):
    (tmp_path / "Snakefile").write_text(
        'rule all:\n    input:\n        "done"\n\n'
        'rule work:\n    output:\n        "done"\n    shell:\n        "touch done"\n'
    )
    nodes, edges = dag_graph(tmp_path, "dag")
    by_id = {n["id"]: n for n in nodes}
    # An edge points dependency -> dependent; the `dot` layout must place the
    # dependent lower on screen (larger y after the y-flip transform).
    source, target = edges[0]["source"], edges[0]["target"]
    assert by_id[target]["position"]["y"] > by_id[source]["position"]["y"]


def test_snakemake_dot_reports_missing_binary(tmp_path, monkeypatch):
    def _no_binary(*args, **kwargs):
        raise FileNotFoundError("snakemake")

    monkeypatch.setattr("neofoam.tooling.workflow.dag.subprocess.run", _no_binary)
    with pytest.raises(RuntimeError, match="not installed"):
        snakemake_dot(tmp_path, "dag")


def test_dag_graph_falls_back_to_layered_layout_without_dot(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "neofoam.tooling.workflow.dag.snakemake_dot", lambda *a, **k: _DOT
    )
    # No `dot` binary → _graphviz_layout returns None and dag_graph uses the
    # pure-Python layered fallback, which still orders dependents below deps.
    monkeypatch.setattr("neofoam.tooling.workflow.dag.shutil.which", lambda name: None)
    nodes, _ = dag_graph(tmp_path, "dag")
    by_id = {n["id"]: n for n in nodes}
    assert (
        by_id["0"]["position"]["y"]
        > by_id["1"]["position"]["y"]
        > by_id["2"]["position"]["y"]
    )


def test_parse_dot_falls_back_on_malformed_color():
    dot = 'digraph {\n\t0[label = "x", color = "garbage", style="rounded"];\n}\n'
    nodes, _ = parse_dot(dot)
    assert nodes["0"]["color"] == "#888888"
