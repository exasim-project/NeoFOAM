# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Canvas model for the wizard's parameter-sweep step (VueFlow nodes + edges).

The presentation layer of the sweep: it maps sweep *dimensions* and the packaged
rule library onto plain node/edge dicts (positions, handles, edge styles) the
trame-flow canvas renders, and reads the canvas back into a
``{dim: {variant: payload}}`` mapping (:func:`nodes_to_dimensions`). No trame
imports — the shapes are plain JSON so this stays exercisable headless.

Validation lives in :mod:`neofoam.tooling.workflow.sweep._validate`; Snakefile
generation in :mod:`~neofoam.tooling.workflow.sweep._codegen`; the export/load
round-trip in :mod:`~neofoam.tooling.workflow.sweep._io`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from neofoam.tooling.workflow.rules import (
    CAD_DIM,
    MESH_DIM,
    RuleKind,
    RuleRegistry,
    RuleSpec,
    SETUP_CONFIG_PATTERN,
    default_registry,
)

#: Canvas node types rendered by the app's CustomNode templates.
DIM_NODE_TYPE = "dim"
RULE_NODE_TYPE = "rule"

#: Anchor rules of every pipeline (the full set lives in the rule registry).
SETUP_RULE = "setup"
SOLVE_RULE = "solve"
ALL_RULE = "all"

#: Handle-id prefixes; both edge ends carry the same normalized payload so a
#: pure string comparison can validate connections.
CFG_HANDLE_PREFIX = "cfg:"
IN_HANDLE_PREFIX = "in:"
OUT_HANDLE_PREFIX = "out:"

#: Edge style for file (rule output -> rule input) edges: solid, bold, blue.
FILE_EDGE_PROPS: dict[str, Any] = {"style": {"strokeWidth": 2.5, "stroke": "#2563eb"}}

#: Edge style for config (dimension -> rule) edges: thin, dashed, muted.
CFG_EDGE_PROPS: dict[str, Any] = {
    "style": {"strokeDasharray": "6 4", "stroke": "#94a3b8", "strokeWidth": 1.5}
}

_WILDCARD_RE = re.compile(r"\{(\w+)\}")


def _normalize_pattern(pattern: str) -> str:
    """Replace every ``{name}`` wildcard with the fixed token ``{*}``."""
    return _WILDCARD_RE.sub("{*}", pattern)


# ---------------------------------------------------------------------------
# Dimensions and canvas nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepDimension:
    """One sweepable dimension: a solver config with named variants.

    Attributes:
        name: The snake-case config name (``transport_properties_config``) —
            the sweep column and ``params.yaml`` key.
        title: Human title for the node header (e.g. ``transportProperties``).
        schema: The config's JSONForms-ready schema (rendered inside the node).
        seed: The initial "base" variant payload (the live wizard form state).
    """

    name: str
    title: str
    schema: dict[str, Any]
    seed: dict[str, Any]


def dim_node(
    node_id: str,
    dim: SweepDimension,
    entries: dict[str, dict[str, Any]] | None = None,
    selected: str | None = None,
    x: float = 0.0,
    y: float = 0.0,
) -> dict[str, Any]:
    """A dimension node: a view into one ``params.yaml`` section.

    ``data.entries`` holds the named variants (``{name: payload}``),
    ``data.selected`` the variant shown in the node, ``data.rename`` the
    uncommitted rename buffer. All values are plain JSON-serializable dicts.
    """
    if entries is None:
        entries = {"base": dict(dim.seed)}
    if selected is None:
        selected = next(iter(entries))
    return {
        "id": node_id,
        "type": DIM_NODE_TYPE,
        "position": {"x": x, "y": y},
        "class": "vue-flow__node-default",
        # The default-node class fixes width at 150px and pads the content;
        # the node template brings its own width and padding.
        "style": {"width": "420px", "padding": "0"},
        "width": "auto",
        "height": "auto",
        "data": {
            "label": dim.title,
            "dim": dim.name,
            "schema": dim.schema,
            "entries": entries,
            "selected": selected,
            "rename": selected,
        },
    }


def rule_nodes(
    dims: Sequence[str],
    registry: RuleRegistry | None = None,
    enabled: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """The pipeline nodes derived from the rule registry's validated plan.

    Args:
        dims: The dimension names currently on the canvas. Non-mesh dims become
            ``setup``'s config ports; a ``mesh`` dim is ``setup_mesh``'s port.
        registry: The rule library (default: :func:`default_registry`).
        enabled: The selected rule names (default: the full default pipeline).
    """
    plan = (registry or default_registry()).plan(enabled)
    setup_dims = sorted(d for d in dims if d not in (MESH_DIM, CAD_DIM))

    def resolve(spec: RuleSpec) -> tuple[list[str], list[str], list[str]]:
        """(cfg_dims, inputs, outputs) with plan-dependent patterns rendered."""
        if spec.kind is RuleKind.AGGREGATE:
            return [], [plan.final_pattern], []
        if spec.kind is RuleKind.MESH_STAGE:
            return [MESH_DIM], list(spec.inputs), list(spec.outputs)
        if spec.is_mesh_tool:
            inputs = [f"meshes/{{mesh}}/{plan.mesh_tool_input[spec.name]}"]
            return [], inputs, list(spec.outputs)
        if spec.kind is RuleKind.CASE_SETUP:
            inputs = [SETUP_CONFIG_PATTERN, f"meshes/{{mesh}}/{plan.mesh_done}"]
            return setup_dims, inputs, list(spec.outputs)
        return [], list(spec.inputs), list(spec.outputs)

    nodes: list[dict[str, Any]] = []
    # Two rows, left-to-right in pipeline order: the keyed mesh chain on top,
    # the per-case chain (setup -> solve [-> post] -> all) below it.
    ordered = [s for s in plan.rules if s.name != ALL_RULE]
    ordered.append(plan.get(ALL_RULE))
    row_index = {MESH_DIM: 0, "case": 0}
    for spec in ordered:
        cfg_dims, inputs, outputs = resolve(spec)
        row = MESH_DIM if spec.keyed_by == MESH_DIM else "case"
        i = row_index[row]
        row_index[row] += 1
        position = {
            "x": (620.0 if row == MESH_DIM else 900.0) + 300.0 * i,
            "y": 40.0 if row == MESH_DIM else 280.0,
        }
        nodes.append(
            {
                "id": f"rule:{spec.name}",
                "type": RULE_NODE_TYPE,
                "position": position,
                "class": "vue-flow__node-default",
                "style": {"width": "auto", "padding": "0"},
                "width": "auto",
                "height": "auto",
                "data": {
                    "label": spec.title or spec.name,
                    "rule": spec.name,
                    "cfg_dims": cfg_dims,
                    "inputs": inputs,
                    "outputs": outputs,
                },
            }
        )
    return nodes


def autowire(nodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wire up the canvas: file-pattern edges plus dimension -> rule cfg edges.

    File edges connect matching output/input patterns across the rule nodes;
    config edges connect every dimension node to every rule node that lists its
    dimension in ``cfg_dims``. Returns the edge list (assign it to
    ``NodeEditor.edges``).
    """
    rules = [n for n in nodes if n.get("type") == RULE_NODE_TYPE]
    dims = [n for n in nodes if n.get("type") == DIM_NODE_TYPE]

    producers: dict[str, tuple[str, str]] = {}
    for node in rules:
        for out in node["data"]["outputs"]:
            producers[_normalize_pattern(out)] = (node["id"], out)

    edges: list[dict[str, Any]] = []
    for node in rules:
        for infile in node["data"]["inputs"]:
            hit = producers.get(_normalize_pattern(infile))
            if hit is None or hit[0] == node["id"]:
                continue
            source_id, out_pattern = hit
            edges.append(
                {
                    "id": f"{source_id}->{node['id']}:{infile}",
                    "source": source_id,
                    "target": node["id"],
                    "sourceHandle": f"{OUT_HANDLE_PREFIX}{out_pattern}",
                    "targetHandle": f"{IN_HANDLE_PREFIX}{infile}",
                    "type": "default",
                    **FILE_EDGE_PROPS,
                }
            )

    for dim in sorted(dims, key=lambda n: str(n["data"]["dim"])):
        cfg_id = f"{CFG_HANDLE_PREFIX}{dim['data']['dim']}"
        for node in rules:
            if dim["data"]["dim"] not in node["data"]["cfg_dims"]:
                continue
            edges.append(
                {
                    "id": f"{dim['id']}->{node['id']}:{cfg_id}",
                    "source": dim["id"],
                    "target": node["id"],
                    "sourceHandle": cfg_id,
                    "targetHandle": cfg_id,
                    "type": "default",
                    **CFG_EDGE_PROPS,
                }
            )
    return edges


def nodes_to_dimensions(
    nodes: Sequence[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract ``{dim: {variant: payload}}`` from the canvas's dimension nodes.

    Raises:
        ValueError: On a dimension node without variants, an empty variant
            name, or a variant name repeated across nodes of one dimension.
    """
    dimensions: dict[str, dict[str, dict[str, Any]]] = {}
    for node in nodes:
        if node.get("type") != DIM_NODE_TYPE:
            continue
        data = node.get("data", {})
        dim = str(data.get("dim", ""))
        entries = data.get("entries") or {}
        if not entries:
            msg = f"node '{node['id']}' ({dim}): no variants defined"
            raise ValueError(msg)
        variants = dimensions.setdefault(dim, {})
        for raw_name, payload in entries.items():
            name = str(raw_name).strip()
            if not name:
                msg = f"node '{node['id']}' ({dim}): the variant name must not be empty"
                raise ValueError(msg)
            if name in variants:
                msg = f"duplicate variant name '{name}' for dimension '{dim}'"
                raise ValueError(msg)
            variants[name] = dict(payload or {})
    return dimensions
