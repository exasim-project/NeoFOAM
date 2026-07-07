# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The Parameters step's data model — pure Python, no trame, no VueFlow.

``SweepModel`` owns the sweep definition (dimensions → named variants → full
config payloads, the picked display fields, the enabled pipeline rules and a
dirty flag) and every derivation over it: the case cross product, the parameters
table, live validation and the variant-series generator. ``SweepPanel`` holds
one of these and renders it into trame state + canvas nodes; keeping the logic
here makes it unit-testable with no server (see ``test/ui/test_sweep_model.py``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from neofoam.workflow.rules import DEFAULT_ENABLED, MESH_DIM
from neofoam.workflow.sweep import cross_product, variant_errors

__all__ = [
    "DimensionState",
    "Overview",
    "SweepModel",
    "series_values",
    "variant_name",
]

#: Variant names become case names and directory components.
_VARIANT_NAME_RE = re.compile(r"[A-Za-z0-9_.\-]+")


def series_values(
    mode: str, values_text: str, vmin: str, vmax: str, count: str
) -> list[float]:
    """The variant value series for the generator.

    ``list`` parses comma/space-separated numbers; ``linear``/``log`` span
    ``count`` values from ``vmin`` to ``vmax`` (inclusive), rounded to 10
    significant digits so generated payloads and names stay readable.

    Raises:
        ValueError: With a user-facing message on any invalid input.
    """
    if mode == "list":
        tokens = [t for t in re.split(r"[\s,;]+", values_text.strip()) if t]
        if not tokens:
            msg = "give at least one value"
            raise ValueError(msg)
        values = []
        for token in tokens:
            try:
                values.append(float(token))
            except ValueError:
                msg = f"'{token}' is not a number"
                raise ValueError(msg) from None
        return values
    try:
        lo, hi, n = float(vmin), float(vmax), int(count)
    except (TypeError, ValueError):
        msg = "min, max and count must be numbers"
        raise ValueError(msg) from None
    if n < 2:
        msg = "count must be at least 2"
        raise ValueError(msg)
    if mode == "linear":
        step = (hi - lo) / (n - 1)
        raw = [lo + i * step for i in range(n)]
    elif mode == "log":
        if lo <= 0 or hi <= 0:
            msg = "log spacing needs positive min and max"
            raise ValueError(msg)
        ratio = (hi / lo) ** (1.0 / (n - 1))
        raw = [lo * ratio**i for i in range(n)]
    else:
        msg = f"unknown spacing mode '{mode}'"
        raise ValueError(msg)
    return [float(f"{v:.10g}") for v in raw]


def variant_name(param: str, value: float | int) -> str:
    """An auto variant name like ``nu1e-05`` (``+`` stripped: not name-safe)."""
    return f"{param}{value:g}".replace("+", "")


def _varying_keys(variants: dict[str, dict[str, Any]]) -> list[str]:
    """The payload keys whose values differ between a dimension's variants."""
    keys = sorted({k for payload in variants.values() for k in payload})
    return [
        k
        for k in keys
        if len({json.dumps(p.get(k), sort_keys=True) for p in variants.values()}) > 1
    ]


def _cell(value: Any) -> Any:
    """A variant payload value as a table cell (nested payloads as JSON text)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value)


@dataclass
class DimensionState:
    """One sweep dimension: a config with named variants and a display slice.

    ``entries`` holds the *full* variant payloads (unrendered keys survive edits
    on a sliced dimension); ``fields`` is the picked display slice (empty = the
    whole form); ``schema`` is the JSONForms schema actually rendered (sliced to
    ``fields`` when picked). ``selected`` / ``rename`` are the active variant and
    the rename buffer.

    ``kind`` is ``"config"`` for a solver-config dimension (validated against a
    pydantic class), ``"cad"`` for the CAD geometry axis (a numeric parameter map
    validated only structurally), or ``"mesh"`` for the reserved keyed mesh
    dimension (variants are config-name-keyed ``{config_name: payload}`` maps, one
    per mesh dict source, validated against each source's class);
    ``model_path`` is the parametric model file for a ``cad`` dimension (empty
    otherwise).
    """

    name: str
    title: str
    schema: dict[str, Any]
    entries: dict[str, dict[str, Any]]
    selected: str
    fields: list[str] = field(default_factory=list)
    rename: str = ""
    kind: str = "config"
    model_path: str = ""

    def numeric_params(self) -> list[dict[str, Any]]:
        """The top-level numeric properties (the series generator's targets)."""
        return [
            {"title": prop.get("title", key), "value": key, "type": prop.get("type")}
            for key, prop in self.schema.get("properties", {}).items()
            if isinstance(prop, dict) and prop.get("type") in ("number", "integer")
        ]

    def _is_integer(self, param: str) -> bool:
        prop = self.schema.get("properties", {}).get(param, {})
        return isinstance(prop, dict) and prop.get("type") == "integer"


@dataclass
class Overview:
    """Everything the panel needs to render after a mutation (pure derivation)."""

    headers: list[dict[str, str]]
    rows: list[dict[str, Any]]
    case_count: int
    count_label: str
    warn: bool
    valid: bool
    errors: dict[str, dict[str, str]]


class SweepModel:
    """The sweep definition: dimensions, enabled rules, dirty flag + derivations."""

    def __init__(self) -> None:
        self.dims: dict[str, DimensionState] = {}
        self.enabled: list[str] = list(DEFAULT_ENABLED)
        self.dirty: bool = False

    # -- queries ---------------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self.dims

    def sorted_dims(self) -> list[DimensionState]:
        return [self.dims[name] for name in sorted(self.dims)]

    def to_dimensions(self) -> dict[str, dict[str, dict[str, Any]]]:
        """The ``{dim: {variant: payload}}`` mapping consumed by export."""
        return {name: dict(d.entries) for name, d in self.dims.items()}

    # -- structural mutations --------------------------------------------------

    def add_dimension(
        self,
        name: str,
        *,
        title: str,
        schema: dict[str, Any],
        seed: dict[str, Any],
        fields: list[str] | None = None,
    ) -> None:
        """Add a dimension seeded with a single ``base`` variant.

        Raises:
            ValueError: If a dimension of this name is already present.
        """
        if name in self.dims:
            msg = f"'{title}' is already on the canvas."
            raise ValueError(msg)
        self.dims[name] = DimensionState(
            name=name,
            title=title,
            schema=schema,
            entries={"base": dict(seed)},
            selected="base",
            fields=list(fields or []),
            rename="base",
        )
        self.dirty = True

    def add_cad_dimension(
        self,
        name: str,
        *,
        title: str,
        model_path: str,
        params: dict[str, float],
    ) -> None:
        """Add the CAD geometry axis, seeded with a single ``base`` variant.

        A CAD dimension is a numeric parameter map ``{alias: value}`` driving a
        parametric model; its schema is synthesised as a flat numeric JSONForms
        object (no solver config class). It is exempt from config validation.

        Raises:
            ValueError: If a dimension of this name is already present.
        """
        if name in self.dims:
            msg = f"'{title}' is already on the canvas."
            raise ValueError(msg)
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                alias: {"type": "number", "title": alias} for alias in params
            },
        }
        self.dims[name] = DimensionState(
            name=name,
            title=title,
            schema=schema,
            entries={"base": dict(params)},
            selected="base",
            fields=list(params),
            rename="base",
            kind="cad",
            model_path=model_path,
        )
        self.dirty = True

    def cad_dimensions(self) -> dict[str, dict[str, Any]]:
        """The CAD axes as ``{name: {"model": path, "variants": {name: params}}}``.

        The shape :func:`neofoam.workflow.sweep.export_sweep` consumes as its
        ``cad`` kwarg; empty when no CAD dimension is on the canvas.
        """
        return {
            name: {"model": d.model_path, "variants": dict(d.entries)}
            for name, d in self.dims.items()
            if d.kind == "cad"
        }

    def add_mesh_source(
        self,
        config_name: str,
        *,
        title: str,
        schema: dict[str, Any],
        seed: dict[str, Any],
    ) -> None:
        """Add a mesh dict (blockMesh/snappy) as a source of the keyed ``mesh`` dim.

        The reserved ``mesh`` dimension is *config-name-keyed*: every variant
        payload maps each source's config name to that source's full config
        (``{"block_mesh_dict_config": {...}, "snappy_hex_mesh_dict_config":
        {...}}``). Adding the first source creates the dimension (seeded with a
        single ``base`` variant); a further source is grafted onto *every*
        existing variant (seeded from ``seed``), and its schema property joins the
        combined mesh form so both configs render together.

        Raises:
            ValueError: If this config is already a mesh source.
        """
        prop = {**schema, "title": title}
        mesh = self.dims.get(MESH_DIM)
        if mesh is None:
            self.dims[MESH_DIM] = DimensionState(
                name=MESH_DIM,
                title="Mesh",
                schema={"type": "object", "properties": {config_name: prop}},
                entries={"base": {config_name: dict(seed)}},
                selected="base",
                rename="base",
                kind="mesh",
            )
            self.dirty = True
            return
        if config_name in mesh.schema.get("properties", {}):
            msg = f"'{title}' is already a mesh source."
            raise ValueError(msg)
        mesh.schema = {
            **mesh.schema,
            "properties": {**mesh.schema.get("properties", {}), config_name: prop},
        }
        for payload in mesh.entries.values():
            payload.setdefault(config_name, dict(seed))
        self.dirty = True

    def remove_mesh_source(self, config_name: str) -> None:
        """Drop one mesh source; remove the whole ``mesh`` dim if it was the last.

        Raises:
            ValueError: If there is no mesh dimension or this config is not a
                source of it.
        """
        mesh = self.dims.get(MESH_DIM)
        if mesh is None or config_name not in mesh.schema.get("properties", {}):
            msg = f"'{config_name}' is not a mesh source."
            raise ValueError(msg)
        props = {
            k: v
            for k, v in mesh.schema.get("properties", {}).items()
            if k != config_name
        }
        if not props:
            del self.dims[MESH_DIM]
            self.dirty = True
            return
        mesh.schema = {**mesh.schema, "properties": props}
        for payload in mesh.entries.values():
            payload.pop(config_name, None)
        self.dirty = True

    def mesh_sources(self) -> list[str]:
        """The config names currently sourced by the ``mesh`` dimension (sorted)."""
        mesh = self.dims.get(MESH_DIM)
        if mesh is None:
            return []
        return sorted(mesh.schema.get("properties", {}))

    def remove_dimension(self, name: str) -> None:
        """Drop a dimension and its variants.

        Raises:
            ValueError: If the dimension is not present.
        """
        if name not in self.dims:
            msg = f"'{name}' is not on the canvas."
            raise ValueError(msg)
        del self.dims[name]
        self.dirty = True

    def set_enabled(self, enabled: list[str]) -> None:
        self.enabled = list(enabled)
        self.dirty = True

    def load(self, dims: list[DimensionState], enabled: list[str]) -> None:
        """Replace the whole definition (round-trip load); starts clean."""
        self.dims = {d.name: d for d in dims}
        self.enabled = list(enabled)
        self.dirty = False

    def mark_exported(self) -> None:
        self.dirty = False

    # -- variant mutations -----------------------------------------------------

    def variant_add(self, name: str) -> str:
        """Clone the selected variant into a fresh ``variant-N``; return its name."""
        dim = self.dims[name]
        n = 1
        while f"variant-{n}" in dim.entries:
            n += 1
        new = f"variant-{n}"
        dim.entries[new] = dict(dim.entries[dim.selected])
        dim.selected = dim.rename = new
        self.dirty = True
        return new

    def variant_delete(self, name: str) -> None:
        """Delete the selected variant.

        Raises:
            ValueError: If it is the dimension's only variant.
        """
        dim = self.dims[name]
        if len(dim.entries) <= 1:
            msg = f"'{dim.title}' needs at least one variant."
            raise ValueError(msg)
        dim.entries.pop(dim.selected, None)
        dim.selected = dim.rename = next(iter(dim.entries))
        self.dirty = True

    def variant_select(self, name: str, variant: str) -> None:
        dim = self.dims[name]
        if variant in dim.entries:
            dim.selected = dim.rename = variant

    def variant_rename(self, name: str, new: str) -> None:
        """Rename the selected variant.

        Raises:
            ValueError: On an invalid or duplicate name.
        """
        dim = self.dims[name]
        old, new = dim.selected, new.strip()
        if new == old:
            return
        if not _VARIANT_NAME_RE.fullmatch(new) or new in dim.entries:
            msg = (
                f"Invalid variant name '{new}' — use letters, digits, '_', '-', '.'"
                " and keep names unique."
            )
            raise ValueError(msg)
        dim.entries = {(new if k == old else k): v for k, v in dim.entries.items()}
        dim.selected = dim.rename = new
        self.dirty = True

    def set_rename_buffer(self, name: str, value: str) -> None:
        self.dims[name].rename = value

    def variant_edit(self, name: str, payload: dict[str, Any]) -> None:
        """Apply a form edit to the selected variant.

        On a sliced dimension the edit *merges* (only the picked fields are
        rendered, so the unrendered keys of the full payload must survive); on a
        whole-form dimension it replaces.
        """
        dim = self.dims[name]
        if dim.fields:
            dim.entries[dim.selected] = {**dim.entries[dim.selected], **payload}
        else:
            dim.entries[dim.selected] = dict(payload)
        self.dirty = True

    def generate_series(
        self, name: str, param: str, values: list[float], *, replace: bool
    ) -> int:
        """Create one variant per value; return the number generated."""
        dim = self.dims[name]
        template = dict(dim.entries[dim.selected])
        if replace:
            dim.entries = {}
        for value in values:
            typed: float | int = int(round(value)) if dim._is_integer(param) else value
            base = variant_name(param, typed)
            candidate, n = base, 2
            while candidate in dim.entries:
                candidate, n = f"{base}-{n}", n + 1
            dim.entries[candidate] = {**template, param: typed}
        dim.selected = dim.rename = next(iter(dim.entries))
        self.dirty = True
        return len(values)

    # -- derivations -----------------------------------------------------------

    def overview(self, classes: dict[str, Any], warn_threshold: int) -> Overview:
        """Case count, factorized label, validation and the parameters table."""
        dimensions = self.to_dimensions()
        try:
            rows = cross_product(dimensions) if dimensions else []
        except ValueError:
            dimensions, rows = {}, []
        # CAD variants are numeric parameter maps with no solver config class, so
        # they are exempt from config validation (they are always structurally
        # valid); they still flow through the table + case count unchanged.
        validated = {d: v for d, v in dimensions.items() if self.dims[d].kind != "cad"}
        errors = variant_errors(validated, classes)

        # Factorized count: "2 × 3 = 6 case(s)".
        per_dim = [len(dimensions[d]) for d in sorted(dimensions)]
        product = 1
        for n in per_dim:
            product *= n
        if len(per_dim) > 1:
            label = f"{' × '.join(str(n) for n in per_dim)} = {product} case(s)"
        else:
            label = f"{product if per_dim else 0} case(s)"

        headers, rows = self._table(dimensions, rows, errors)
        return Overview(
            headers=headers,
            rows=rows,
            case_count=len(rows),
            count_label=label,
            warn=len(rows) > warn_threshold,
            valid=not errors,
            errors=errors,
        )

    def _table(
        self,
        dimensions: dict[str, dict[str, dict[str, Any]]],
        rows: list[dict[str, Any]],
        errors: dict[str, dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        varying = {dim: _varying_keys(dimensions[dim]) for dim in sorted(dimensions)}
        # Column titles are bare parameter names; qualify only on a cross-dim
        # collision (two dimensions vary the same key).
        counts: dict[str, int] = {}
        for keys in varying.values():
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
        headers: list[dict[str, str]] = [{"title": "case", "key": "case"}]
        for dim, keys in varying.items():
            title = self.dims[dim].title if dim in self.dims else dim
            headers.append({"title": title, "key": dim})
            for key in keys:
                column = key if counts[key] == 1 else f"{title} · {key}"
                # "__" not "." — a dotted key is a VDataTable nested-object path.
                headers.append({"title": column, "key": f"{dim}__{key}"})
                for row in rows:
                    row[f"{dim}__{key}"] = _cell(dimensions[dim][row[dim]].get(key))
        for row in rows:
            msgs = [
                f"{dim}.{row[dim]}: {errors[dim][row[dim]]}"
                for dim in dimensions
                if row[dim] in errors.get(dim, {})
            ]
            row["validation"] = "; ".join(msgs) if msgs else "✓"
        if rows:
            headers.append({"title": "validation", "key": "validation"})
        return (headers if rows else []), rows

    def variant_error(self, name: str, classes: dict[str, Any]) -> str:
        """The selected variant's validation message (for the Configure alert)."""
        dim = self.dims.get(name)
        if dim is None or dim.kind == "cad":
            return ""
        errors = variant_errors(
            {name: {dim.selected: dim.entries[dim.selected]}}, classes
        )
        return errors.get(name, {}).get(dim.selected, "")
