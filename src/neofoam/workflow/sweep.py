# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""UI-free data layer for the case wizard's parameter-sweep canvas.

Maps a trame-flow canvas (plain node/edge dicts) to a Snakemake parameter
sweep over a saved base case:

- A *dimension* node is a live view into one ``params.yaml`` section — one
  solver config (e.g. ``transport_properties_config``) holding named variants.
  The reserved ``mesh`` dimension is *keyed*: its variants carry config-name-
  keyed payloads (``{"block_mesh_dict_config": {...}}``) and the mesh rules run
  once per variant under ``meshes/{variant}/``, shared by every case.
- The pipeline comes from the packaged rule library
  (:mod:`neofoam.workflow.rules`): ``setup_mesh`` → per-tool mesh rules
  (blockMesh/snappyHexMesh/checkMesh) once per mesh variant, then per case
  ``setup`` (clone base + apply configs + copy the variant's mesh) → ``solve``
  (``--no-preprocess``) → ``all``.
- :func:`autowire` connects dimension nodes to their consuming rules and the
  rules to each other by file pattern.
- :func:`export_sweep` reads the canvas back and writes a runnable workflow
  directory (``sweep.csv``, ``params.yaml``, ``Snakefile``, ``configs/``). The
  generated Snakefile is a thin header that ``include:``\\ s the packaged
  ``.smk`` files and shells out to ``python -m neofoam.workflow.sweep_runner``
  — no module copies needed.

This module deliberately has no trame imports so it can be exercised headless.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from neofoam.workflow.paramspace import (
    YamlParamSpace,
    read_params_yaml,
    write_if_changed,
    write_params_yaml,
    write_sweep_csv,
)
from neofoam.workflow.rules import (
    CAD_DIM,
    DEFAULT_ENABLED,
    MESH_DIM,
    RuleRegistry,
    RuleSpec,
    SETUP_CONFIG_PATTERN,
    default_registry,
)

#: Rule that regenerates the STLs when a CAD axis is swept (opt-in, like post).
CAD_RULE = "cad_geometry"

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
        if spec.name == ALL_RULE:
            return [], [plan.final_pattern], []
        if spec.consumes_mesh_dim:
            return [MESH_DIM], list(spec.inputs), list(spec.outputs)
        if spec.is_mesh_tool:
            inputs = [f"meshes/{{mesh}}/{plan.mesh_tool_input[spec.name]}"]
            return [], inputs, list(spec.outputs)
        if spec.consumes_case_dims:
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


# ---------------------------------------------------------------------------
# Canvas -> sweep -> generated workflow directory
# ---------------------------------------------------------------------------


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


def cross_product(
    dimensions: Mapping[str, Mapping[str, Any]],
    case_col: str = "case",
) -> list[dict[str, str]]:
    """The full cross product of variant names per dimension (sweep.csv rows).

    Case names join the variant names in sorted-dimension order.
    """
    dim_order = sorted(dimensions)
    names_per_dim = [sorted(dimensions[dim]) for dim in dim_order]
    rows: list[dict[str, str]] = []
    seen: dict[str, tuple[str, ...]] = {}
    for combo in itertools.product(*names_per_dim):
        case = "_".join(combo) or "default"
        if case in seen:
            msg = (
                f"case-name collision: '{case}' produced by more than one variant "
                f"combination ({seen[case]} and {combo}) — rename a variant to "
                f"avoid '_' ambiguity"
            )
            raise ValueError(msg)
        seen[case] = combo
        row = {case_col: case}
        row.update(zip(dim_order, combo))
        rows.append(row)
    return rows


def validate_dimensions(
    dimensions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    classes: Mapping[str, type[BaseModel]],
) -> None:
    """Validate every variant payload against its dimension's config class.

    Payloads are validated, not round-tripped: several configs materialize
    their content in a wrap serializer (``model_dump()`` of the defaults is
    ``{}``), so dumping the validated instance would lose the form values.
    The runner re-validates (and thereby coerces) payloads at apply time.

    Raises:
        ValueError: On an unknown dimension or a failing variant, naming the
            offending ``dim.variant``.
    """
    for dim, variants in dimensions.items():
        cls = classes.get(dim)
        if cls is None:
            msg = f"unknown dimension '{dim}' (known: {sorted(classes)})"
            raise ValueError(msg)
        for name, payload in variants.items():
            try:
                cls.model_validate(dict(payload))
            except Exception as e:
                msg = f"variant '{dim}.{name}' failed validation against {cls.__name__}: {e}"
                raise ValueError(msg) from e


def validate_mesh_dimension(
    variants: Mapping[str, Mapping[str, Any]],
    classes: Mapping[str, type[BaseModel]],
) -> None:
    """Validate the ``mesh`` dimension's config-name-keyed variant payloads.

    Each mesh variant holds ``{config_name: payload}`` (e.g.
    ``{"block_mesh_dict_config": {...}}``); every inner payload is validated
    against its config class. Empty variants are allowed (the base dicts run
    unchanged).

    Raises:
        ValueError: Naming the offending ``mesh.variant.config``.
    """
    for name, payloads in variants.items():
        if not isinstance(payloads, Mapping):
            msg = f"mesh variant '{name}' must map config names to payloads"
            raise ValueError(msg)
        for config_name, payload in payloads.items():
            cls = classes.get(str(config_name))
            if cls is None:
                msg = (
                    f"mesh variant '{name}' names unknown config '{config_name}' "
                    f"(known: {sorted(classes)})"
                )
                raise ValueError(msg)
            try:
                cls.model_validate(dict(payload))
            except Exception as e:
                msg = (
                    f"mesh variant '{name}.{config_name}' failed validation "
                    f"against {cls.__name__}: {e}"
                )
                raise ValueError(msg) from e


def validate_cad_dimension(variants: Mapping[str, Any]) -> None:
    """Validate the ``cad`` dimension's numeric parameter-map variants.

    A CAD variant is a plain ``{alias: number}`` map (the parametric model's
    driven dimensions); there is no config class to validate against, only the
    numeric shape.

    Raises:
        ValueError: Naming the offending ``cad.variant`` (or its alias) when a
            variant is not a mapping or carries a non-numeric value.
    """
    for name, payload in variants.items():
        if not isinstance(payload, Mapping):
            msg = f"cad variant '{name}' must map parameter aliases to numbers"
            raise ValueError(msg)
        for alias, value in payload.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = (
                    f"cad variant '{name}.{alias}' must be a number, got "
                    f"{type(value).__name__}"
                )
                raise ValueError(msg)


def _short_validation_error(exc: Exception) -> str:
    """A one-line, user-facing message from a pydantic ``ValidationError``.

    Takes the first error's location + message (e.g. ``nu: Input should be
    greater than 0``); falls back to the exception text for non-pydantic errors.
    """
    if isinstance(exc, ValidationError):
        detail = exc.errors()
        if detail:
            first = detail[0]
            loc = ".".join(str(p) for p in first.get("loc", ())) or "value"
            return f"{loc}: {first.get('msg', 'invalid')}"
    return str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__


def variant_errors(
    dimensions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    classes: Mapping[str, type[BaseModel]],
) -> dict[str, dict[str, str]]:
    """Per-variant validation errors, ``{dim: {variant: message}}``.

    Only failing variants appear. Regular dimensions validate each variant
    payload against the dimension's config class; the reserved ``mesh``
    dimension validates its config-name-keyed inner payloads (each against its
    own class). An unknown dimension marks every one of its variants. Unlike
    :func:`validate_dimensions` this never raises — it is the non-throwing form
    used to badge the canvas live.
    """
    errors: dict[str, dict[str, str]] = {}
    for dim, variants in dimensions.items():
        for name, payload in variants.items():
            msg = _variant_error(dim, payload, classes)
            if msg is not None:
                errors.setdefault(dim, {})[name] = msg
    return errors


def _variant_error(
    dim: str,
    payload: Mapping[str, Any],
    classes: Mapping[str, type[BaseModel]],
) -> str | None:
    """Validate one variant payload; return a message or ``None`` if valid."""
    if dim == MESH_DIM:
        if not isinstance(payload, Mapping):
            return "must map config names to payloads"
        for config_name, inner in payload.items():
            cls = classes.get(str(config_name))
            if cls is None:
                return f"unknown config '{config_name}'"
            try:
                cls.model_validate(dict(inner))
            except Exception as exc:
                return f"{config_name}: {_short_validation_error(exc)}"
        return None
    cls = classes.get(dim)
    if cls is None:
        return f"unknown dimension '{dim}'"
    try:
        cls.model_validate(dict(payload))
    except Exception as exc:
        return _short_validation_error(exc)
    return None


def sweep_snakefile(
    solver_name: str,
    base_case: str | Path,
    dims: Sequence[str],
    *,
    cad_model: str | None = None,
    registry: RuleRegistry | None = None,
    enabled: Sequence[str] | None = None,
) -> str:
    """Generate the Snakefile: a thin header + the packaged rule includes.

    The header builds the :class:`~neofoam.workflow.paramspace.YamlParamSpace`,
    materializes the per-case and per-mesh-variant configs at parse time
    (``write_if_changed`` keeps unchanged cases from re-running) and defines
    the plain globals the static ``.smk`` files consume — Snakemake's
    ``include:`` shares the Snakefile's namespace.

    When ``cad_model`` is given the header also defines ``CAD_MODEL`` and the
    keyed ``cad_axis`` the packaged ``cad_geometry.smk`` consumes (that rule's
    include comes first in the mesh chain, ahead of blockMesh).
    """
    plan = (registry or default_registry()).plan(enabled)
    base = str(Path(base_case).resolve())
    setup_dims = sorted(d for d in dims if d not in (MESH_DIM, CAD_DIM))
    includes = "\n".join(
        f'include: str(rules_dir() / "{spec.smk_file}")'
        for spec in plan.rules
        if spec.smk_file  # `all` is emitted inline below
    )
    cad_global = (
        f"\nCAD_MODEL = {json.dumps(cad_model)}" if cad_model is not None else ""
    )
    cad_axis = (
        f'\ncad_axis = space.keyed("{CAD_DIM}", out_dir="configs")'
        if cad_model is not None
        else ""
    )
    # MESH_STEM is the mesh mini-case dir used verbatim as the mesh-chain rules'
    # static `output:` (so it stays a literal wildcard path, never a lambda). It
    # composes cad × mesh into `meshes/{cad}__{mesh}` when a CAD axis is present,
    # and stays `meshes/{mesh}` otherwise; _mesh_case / _mesh_dir_of compute the
    # concrete dirs for the input:/params: lambdas.
    if cad_model is not None:
        mesh_stem_block = (
            'MESH_STEM = "meshes/{cad}__{mesh}"\n'
            "def _mesh_case(wc):\n"
            '    return f"meshes/{wc.cad}__{wc.mesh}"\n'
            "def _mesh_dir_of(wc):\n"
            '    return f"meshes/{cad_axis.of(wc)}__{mesh_axis.of(wc)}"'
        )
    else:
        mesh_stem_block = (
            'MESH_STEM = "meshes/{mesh}"\n'
            "def _mesh_case(wc):\n"
            '    return f"meshes/{wc.mesh}"\n'
            "def _mesh_dir_of(wc):\n"
            '    return f"meshes/{mesh_axis.of(wc)}"'
        )
    return f'''# Generated by NeoFOAM — parameter sweep composed from the packaged rule
# library (neofoam.workflow.rules; see the included .smk files for the rule
# bodies). Run from this directory:  snakemake -n  (dry run),  snakemake -j4
# Requires: the neofoam package (pybFoam) importable and snakemake to execute.
from neofoam.workflow.paramspace import YamlParamSpace
from neofoam.workflow.rules import rules_dir

SOLVER = {json.dumps(solver_name)}
# The CLI subcommand (`neofoam solver <cmd>`) — the solver name, lowercased.
SOLVER_CMD = {json.dumps(solver_name.lower())}
BASE_CASE = {json.dumps(base)}{cad_global}
SETUP_DIMS = {setup_dims!r}
MESH_TOOL_INPUT = {plan.mesh_tool_input!r}
MESH_DONE = "{plan.mesh_done}"
FINAL_PATTERN = "{plan.final_pattern}"

space = YamlParamSpace("sweep.csv", "params.yaml")
space.materialize({{"setup": SETUP_DIMS}}, out_dir="configs")
mesh_axis = space.keyed("{MESH_DIM}", out_dir="configs"){cad_axis}
{mesh_stem_block}
cases = space.cases

{includes}


# The default target lives in the MAIN Snakefile: Snakemake never picks a rule
# from an include:d file as the implicit default target.
rule all:
    input:
        expand(FINAL_PATTERN, case=cases)
'''


def _check_composite_mesh_names(
    dimensions: Mapping[str, Mapping[str, Any]],
) -> None:
    """Guard the composite ``meshes/{cad}__{mesh}`` mini-case keying.

    With a CAD axis the mesh mini-case dir is ``f"{cad}__{mesh}"``; a variant
    name may itself contain ``__``, so distinct ``(cad, mesh)`` pairs could
    collapse to the same directory (and Snakemake would then mis-route the
    wildcard split). Raise before any file is written, mirroring
    :func:`cross_product`'s case-name guard.
    """
    cad_names = sorted(dimensions.get(CAD_DIM, {}))
    mesh_names = sorted(dimensions.get(MESH_DIM, {})) or ["base"]
    seen: dict[str, tuple[str, str]] = {}
    for cad_name in cad_names:
        for mesh_name in mesh_names:
            key = f"{cad_name}__{mesh_name}"
            if key in seen:
                msg = (
                    f"mesh-key collision: '{key}' from more than one cad/mesh "
                    f"combination ({seen[key]} and {(cad_name, mesh_name)}) — "
                    f"rename a variant to avoid '__' ambiguity"
                )
                raise ValueError(msg)
            seen[key] = (cad_name, mesh_name)


@dataclass(frozen=True)
class SweepExport:
    """The files written by :func:`export_sweep`."""

    out_dir: Path
    sweep_csv: Path
    params_yaml: Path
    snakefile: Path
    configs: list[Path]


def export_sweep(
    out_dir: str | Path,
    *,
    solver_name: str,
    base_case: str | Path,
    dimensions: Mapping[str, dict[str, dict[str, Any]]],
    classes: Mapping[str, type[BaseModel]],
    cad: Mapping[str, Mapping[str, Any]] | None = None,
    registry: RuleRegistry | None = None,
    enabled: Sequence[str] | None = None,
) -> SweepExport:
    """Write the runnable workflow directory for the current canvas.

    Validates every variant (the ``mesh`` dimension via
    :func:`validate_mesh_dimension`, the ``cad`` dimension via
    :func:`validate_cad_dimension`, everything else against its config class),
    builds the cross product and writes ``sweep.csv``, ``params.yaml``, the
    ``Snakefile`` and the materialized configs into ``out_dir`` — per-case
    ``configs/{case}/setup.json`` (mesh + cad excluded: they apply in the mesh
    mini-case) plus per-variant ``configs/mesh/{variant}.json`` (an implicit
    empty ``base`` variant when the sweep has no mesh dimension) and, when a CAD
    axis is present, ``configs/cad/{variant}.json``. The directory is
    immediately runnable with ``snakemake``.

    Args:
        cad: The CAD axis as ``{"cad": {"model": path, "variants": {name:
            params}}}`` (the :meth:`SweepModel.cad_dimensions` shape). Empty or
            ``None`` keeps today's mesh-only behaviour. When given, the
            ``cad_geometry`` rule is enabled and the model path is threaded into
            the Snakefile.
    """
    dimensions = {d: dict(v) for d, v in dimensions.items()}
    info = (cad or {}).get(CAD_DIM) or {}
    variants = info.get("variants")
    if variants:
        dimensions[CAD_DIM] = {k: dict(v) for k, v in variants.items()}
    # A CAD axis exists only once it carries variants — an empty cad dict emits
    # no CAD_MODEL / cad_axis / configs/cad and behaves like a mesh-only sweep.
    has_cad = CAD_DIM in dimensions
    cad_model: str | None = str(info["model"]) if has_cad else None
    if has_cad:
        validate_cad_dimension(dimensions[CAD_DIM])

    case_dims = {d: v for d, v in dimensions.items() if d not in (MESH_DIM, CAD_DIM)}
    validate_dimensions(case_dims, classes)
    if MESH_DIM in dimensions:
        validate_mesh_dimension(dimensions[MESH_DIM], classes)
    if not dimensions:
        msg = "no sweep dimensions on the canvas — add at least one config dimension"
        raise ValueError(msg)
    # Build the sweep rows before creating the output directory so a case-name
    # collision aborts with nothing written to disk (no partial workflow dir).
    rows = cross_product(dimensions)
    if has_cad:
        _check_composite_mesh_names(dimensions)

    # The CAD chain is opt-in (like post): enable it only when a CAD axis is on
    # the canvas so mesh-only sweeps stay untouched.
    plan_enabled: Sequence[str] | None = enabled
    if has_cad:
        base_enabled = list(DEFAULT_ENABLED if enabled is None else enabled)
        if CAD_RULE not in base_enabled:
            base_enabled.append(CAD_RULE)
        plan_enabled = base_enabled

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sweep_path = out / "sweep.csv"
    params_path = out / "params.yaml"
    snakefile_path = out / "Snakefile"

    write_sweep_csv(sweep_path, rows)
    write_params_yaml(params_path, {d: dict(v) for d, v in dimensions.items()})
    write_if_changed(
        snakefile_path,
        sweep_snakefile(
            solver_name,
            base_case,
            sorted(dimensions),
            cad_model=cad_model,
            registry=registry,
            enabled=plan_enabled,
        ),
    )

    space = YamlParamSpace(sweep_path, params_path)
    setup_dims = [d for d in space.dims if d not in (MESH_DIM, CAD_DIM)]
    configs = space.materialize({SETUP_RULE: setup_dims}, out_dir=out / "configs")
    if MESH_DIM in dimensions:
        configs += space.materialize_dims([MESH_DIM], out_dir=out / "configs")
    else:
        implicit = out / "configs" / MESH_DIM / "base.json"
        if write_if_changed(implicit, "{}\n"):
            configs.append(implicit)
    if has_cad:
        configs += space.materialize_dims([CAD_DIM], out_dir=out / "configs")

    return SweepExport(
        out_dir=out,
        sweep_csv=sweep_path,
        params_yaml=params_path,
        snakefile=snakefile_path,
        configs=configs,
    )


@dataclass(frozen=True)
class LoadedSweep:
    """A sweep read back from an exported directory (inverse of export)."""

    solver_name: str
    base_case: str
    dimensions: dict[str, dict[str, dict[str, Any]]]
    enabled: list[str]
    cad_model: str = ""


def load_sweep(
    out_dir: str | Path, *, registry: RuleRegistry | None = None
) -> LoadedSweep:
    """Read an exported sweep directory back into its canvas definition.

    The inverse of :func:`export_sweep`: ``params.yaml`` holds the full
    ``{dimension: {variant: payload}}`` (variants keep their complete payloads),
    and the generated ``Snakefile`` header names the solver, the base case and —
    through its ``include:`` lines plus the inline ``rule all`` — the enabled
    rules (mapped back to names via the rule registry). A sweep exported without
    a mesh dimension has no ``mesh`` key (the implicit ``base`` mesh variant is
    materialized separately, not a canvas dimension), so it round-trips to the
    same canvas.

    Raises:
        ValueError: If ``out_dir`` has no ``params.yaml`` (not an exported
            sweep).
    """
    out = Path(out_dir)
    params_path = out / "params.yaml"
    if not params_path.is_file():
        msg = f"'{out}' is not an exported sweep (no params.yaml)"
        raise ValueError(msg)
    dimensions = read_params_yaml(params_path)
    solver_name, base_case, enabled, cad_model = _read_snakefile_header(
        out / "Snakefile", registry or default_registry()
    )
    return LoadedSweep(
        solver_name=solver_name,
        base_case=base_case,
        dimensions=dimensions,
        enabled=enabled,
        cad_model=cad_model,
    )


def _read_snakefile_header(
    path: Path, registry: RuleRegistry
) -> tuple[str, str, list[str], str]:
    """Recover ``(solver_name, base_case, enabled_rules, cad_model)`` from a Snakefile.

    The enabled rules are the ``include:``d ``.smk`` files mapped to rule names
    via the registry, plus the always-inline ``all`` target. ``cad_model`` is the
    ``CAD_MODEL`` global (empty when the sweep has no CAD axis). Missing/renamed
    header values fall back to sensible defaults so a partial file still loads.
    """
    text = path.read_text() if path.is_file() else ""

    # A full JSON/Python double-quoted string literal, escapes included — the
    # inverse of the ``json.dumps`` the generator emits for these globals.
    quoted = r'("(?:[^"\\]|\\.)*")'

    def match(pattern: str, default: str = "") -> str:
        found = re.search(pattern, text)
        if not found:
            return default
        decoded = json.loads(found.group(1))
        return str(decoded)

    solver_name = match(rf"SOLVER\s*=\s*{quoted}") or "incompressibleFluid"
    base_case = match(rf"BASE_CASE\s*=\s*{quoted}")
    cad_model = match(rf"CAD_MODEL\s*=\s*{quoted}")
    by_smk = {
        registry.get(name).smk_file: name
        for name in registry.names
        if registry.get(name).smk_file
    }
    included = re.findall(r'rules_dir\(\)\s*/\s*"([^"]+)"', text)
    enabled = [by_smk[smk] for smk in included if smk in by_smk]
    # `all` is emitted inline (no include); it is always part of the pipeline.
    if "all" in registry.names and "all" not in enabled:
        enabled.append("all")
    return solver_name, base_case, enabled, cad_model
