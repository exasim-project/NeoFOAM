# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Render an exported sweep's Snakemake graph as canvas nodes and edges.

Snakemake emits its job DAG (or rule graph) as Graphviz DOT (``snakemake
--dag`` / ``--rulegraph``). :func:`dag_graph` runs it over an exported sweep
directory, parses the DOT, lays the graph out (with the Graphviz ``dot``
binary when available, a pure-Python layered fallback otherwise) and returns
plain VueFlow node/edge dicts — the same shape the sweep canvas uses, so the
wizard's Parameters step can display the resulting DAG below the editor.

snakemake is an *execution-time* dependency of exported sweeps, not of
neofoam — a missing binary surfaces as a clear :class:`RuntimeError`.

This module deliberately has no trame imports so it can be exercised headless.
"""

from __future__ import annotations

import colorsys
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["dag_graph", "layered_layout", "parse_dot", "snakemake_dot"]

#: Pixels per Graphviz inch, and the vertical/horizontal spacing of the
#: pure-Python fallback layout.
_SCALE = 110
_LAYER_GAP = 130
_NODE_GAP = 210

#: A layout maps a node id to (x, y, width, height) in VueFlow pixels
#: (top-left origin, y growing downward).
Layout = dict[str, tuple[float, float, float, float]]

_NODE_RE = re.compile(
    r'^\s*(\d+)\[label\s*=\s*"(?P<label>(?:[^"\\]|\\.)*)".*?'
    r'color\s*=\s*"(?P<color>[^"]*)"',
)
_EDGE_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)")


def snakemake_dot(sweep_dir: str | Path, mode: str = "dag") -> str:
    """The Graphviz DOT of ``sweep_dir``'s job ``dag`` or ``rulegraph``.

    Raises:
        RuntimeError: When snakemake is not installed or exits non-zero
            (message carries its stderr).
    """
    if mode not in ("dag", "rulegraph"):
        msg = f"unknown snakemake graph mode '{mode}' (use 'dag' or 'rulegraph')"
        raise ValueError(msg)
    try:
        result = subprocess.run(
            ["snakemake", f"--{mode}"],
            cwd=str(sweep_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        msg = "snakemake is not installed — install it to visualize/run the sweep"
        raise RuntimeError(msg) from exc
    if result.returncode != 0:
        msg = f"snakemake --{mode} failed:\n{result.stderr.strip()}"
        raise RuntimeError(msg)
    return result.stdout


def _hsv_to_hex(color: str) -> str:
    """Snakemake's ``"h s v"`` (each in 0..1) node color as a hex string."""
    try:
        h, s, v = (float(c) for c in color.split())
    except ValueError:
        return "#888888"
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def parse_dot(dot: str) -> tuple[dict[str, dict[str, str]], list[tuple[str, str]]]:
    """Parse snakemake DOT into ``{id: {label, color}}`` and an edge list."""
    nodes: dict[str, dict[str, str]] = {}
    edges: list[tuple[str, str]] = []
    for line in dot.splitlines():
        edge_match = _EDGE_RE.match(line)
        if edge_match:
            edges.append((edge_match.group(1), edge_match.group(2)))
            continue
        node_match = _NODE_RE.match(line)
        if node_match:
            label = node_match.group("label").replace("\\n", "\n")
            nodes[node_match.group(1)] = {
                "label": label,
                "color": _hsv_to_hex(node_match.group("color")),
            }
    return nodes, edges


def _estimate_size(label: str) -> tuple[float, float]:
    """Rough pixel size of a node box from its (possibly multi-line) label."""
    lines = label.split("\n")
    width = max((len(line) for line in lines), default=1) * 8 + 32
    height = len(lines) * 20 + 24
    return width, height


def _graphviz_layout(dot: str) -> Layout | None:
    """Lay the graph out with the ``dot`` binary. ``None`` when it is missing."""
    if shutil.which("dot") is None:
        return None
    plain = subprocess.run(
        ["dot", "-Tplain"],
        input=dot,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    graph_height = 0.0
    raw: dict[str, tuple[float, float, float, float]] = {}
    for line in plain.splitlines():
        parts = line.split()
        if parts and parts[0] == "graph":
            graph_height = float(parts[3])
        elif parts and parts[0] == "node":
            x, y, w, h = (float(p) for p in parts[2:6])
            raw[parts[1]] = (x, y, w, h)

    layout: Layout = {}
    for node_id, (x, y, w, h) in raw.items():
        # Graphviz yields inch coordinates anchored at the centre with y
        # growing upward; VueFlow wants top-left pixels with y growing down.
        px = (x - w / 2) * _SCALE
        py = (graph_height - (y + h / 2)) * _SCALE
        layout[node_id] = (px, py, w * _SCALE, h * _SCALE)
    return layout


def layered_layout(
    nodes: dict[str, dict[str, str]], edges: list[tuple[str, str]]
) -> Layout:
    """Fallback layout: each node sits one layer below its deepest input.

    Edges point from dependency to dependent (``u -> v``), so targets like
    ``all`` end up at the bottom, matching Graphviz.
    """
    parents: dict[str, list[str]] = {n: [] for n in nodes}
    for u, v in edges:
        parents.setdefault(v, []).append(u)
        parents.setdefault(u, parents.get(u, []))

    depth: dict[str, int] = {}

    def resolve(node: str, stack: frozenset[str]) -> int:
        if node in depth:
            return depth[node]
        ps = [p for p in parents.get(node, []) if p not in stack]
        d = 1 + max((resolve(p, stack | {node}) for p in ps), default=-1)
        depth[node] = d
        return d

    for node in nodes:
        resolve(node, frozenset())

    layers: dict[int, list[str]] = {}
    for node in nodes:
        layers.setdefault(depth[node], []).append(node)

    layout: Layout = {}
    for layer, members in layers.items():
        offset = -(len(members) - 1) * _NODE_GAP / 2
        for i, node in enumerate(members):
            w, h = _estimate_size(nodes[node]["label"])
            layout[node] = (offset + i * _NODE_GAP, layer * _LAYER_GAP, w, h)
    return layout


def dag_graph(
    sweep_dir: str | Path, mode: str = "dag"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run snakemake over ``sweep_dir`` and build VueFlow nodes and edges."""
    dot = snakemake_dot(sweep_dir, mode)
    dot_nodes, dot_edges = parse_dot(dot)
    layout = _graphviz_layout(dot) or layered_layout(dot_nodes, dot_edges)

    nodes: list[dict[str, Any]] = []
    for node_id, info in dot_nodes.items():
        x, y, w, h = layout.get(node_id, (0.0, 0.0, *_estimate_size(info["label"])))
        color = info["color"]
        nodes.append(
            {
                "id": node_id,
                "type": "default",
                "position": {"x": x, "y": y},
                "width": round(w),
                "height": round(h),
                "data": {"label": info["label"], "color": color},
                "style": {
                    "border": f"2px solid {color}",
                    "borderRadius": "6px",
                    "background": f"{color}22",
                    "whiteSpace": "pre-line",
                    "fontWeight": "600",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            }
        )

    edges: list[dict[str, Any]] = [
        {
            "id": f"{u}->{v}",
            "source": u,
            "target": v,
            "type": "smoothstep",
            "markerEnd": {"type": "arrowclosed", "width": 18, "height": 18},
        }
        for u, v in dot_edges
    ]
    return nodes, edges
