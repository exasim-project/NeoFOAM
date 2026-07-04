# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Build the JSONForms form descriptors for a solver (headless, pure).

Each :class:`FormEntry` carries the JSON Schema + defaults that drive a client-side
``<json-forms>`` component, plus the trame ``state_key`` under which that form's live
data lives, the owning optional model (for hide/skip) and the wizard step it belongs
to. A ``0/<field>`` config is split into two entries — an *input* half
(``dimensions``/``internalField``) and a *boundary-conditions* half
(``boundaryField``) — mirroring the marimo wizard, so each renders in its own step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from neofoam.agent.case_forms import INPUT_KEYS, field_name, is_scheme_config
from neofoam.framework.solver.configurations import configurations
from neofoam.io.pydantic_schema import slice_schema
from neofoam.mcp import tools

__all__ = [
    "FormEntry",
    "build_forms",
    "humanize",
    "jsonforms_schema",
    "allowed_bc_types",
    "seed_boundary_field",
    "patch_bc_schema",
]

_BC_KEYS = ("boundaryField",)

# Mesh / preprocessing dict configs belong to the upstream meshing stage (the e2e
# geometry→mesh workflow), not the physics case wizard. They also aren't writable via
# write_configs' merged path (preprocess.yaml uses YAMLStrategy). Exclude them.
_MESH_FILES = frozenset({"blockMeshDict", "snappyHexMeshDict", "preprocess.yaml"})


@dataclass(frozen=True)
class FormEntry:
    """One rendered form (a whole dict config, or one half of a field config)."""

    key: str
    """Registry key, e.g. ``"dict:ControlDictConfig"`` / ``"field_bc:UFieldConfig"``."""
    config_name: str
    """Snake-case class name — the key ``tools.save_case`` expects."""
    cls_name: str
    """Config class name, e.g. ``"ControlDictConfig"``."""
    title: str
    """Panel title."""
    schema: dict[str, Any]
    """JSON Schema fed to ``<json-forms :schema>`` (sliced for field halves)."""
    defaults: dict[str, Any]
    """Prefill / initial form data (``:data``)."""
    state_key: str
    """Trame state var holding this form's live data object."""
    kind: str
    """``"dict"`` | ``"field_in"`` | ``"field_bc"``."""
    step: str
    """Wizard step id: ``models`` | ``schemes`` | ``bcs`` | ``initial``."""
    owner_model: str | None
    """Optional model that owns this config (hide/skip); ``None`` ⇒ always shown."""
    cls: Any = field(default=None, compare=False)
    """The config class (used by field-half merge in ``case_spec``)."""
    uischema: dict[str, Any] | None = None
    """Optional JSONForms UISchema (``None`` ⇒ auto-layout from the schema)."""


def _owner_by_cls_name(solver: Any) -> dict[str, str]:
    """Map each *optional*-model-owned config class name → its model name.

    Configs owned only by required models (or by no model) are absent → ``None``
    owner ⇒ never hidden.
    """
    owner: dict[str, str] = {}
    for entry in tools.model_catalog(solver):
        if entry.required:
            continue
        for cls_name in (*entry.dicts, *entry.fields):
            owner[cls_name] = entry.name
    return owner


def _slice_defaults(defaults: dict[str, Any], keep: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in defaults.items() if k in keep}


# camelCase / snake_case boundary for human labels: "writeControl" → "Write control".
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Sphinx roles in pydantic docstrings: ":class:`~a.b.C`" → "C".
_SPHINX_ROLE_RE = re.compile(r":[a-zA-Z:]+:`~?([^`]+)`")


def humanize(key: str) -> str:
    """A property key as a human label: ``deltaT`` → "Delta T", ``nu`` → "Nu"."""
    words = _CAMEL_RE.sub(" ", key.replace("_", " ")).strip()
    return words[:1].upper() + words[1:]


def _label_title(key: str, node: Any) -> Any:
    """Replace a pydantic auto-title (``"Writecontrol"``) with :func:`humanize`.

    A hand-written title (anything that isn't just the key re-capitalised) is kept.
    """
    if not isinstance(node, dict):
        return node
    title = node.get("title")
    auto = title is None or (
        isinstance(title, str) and title.replace(" ", "").lower() == key.lower()
    )
    return {**node, "title": humanize(key)} if auto else node


def _clean_description(text: str) -> str:
    """First paragraph of a docstring description, Sphinx markup stripped."""
    first = text.strip().split("\n\n", 1)[0]
    first = _SPHINX_ROLE_RE.sub(lambda m: m.group(1).rsplit(".", 1)[-1], first)
    first = first.replace("``", "")
    return re.sub(r"\s+", " ", first).strip()


def _const_type_of(arm: dict[str, Any], defs: dict[str, Any]) -> str | None:
    """The ``type`` discriminator ``const`` of a union arm (resolving ``$ref``)."""
    node = arm
    ref = arm.get("$ref")
    if ref:
        node = defs.get(ref.rsplit("/", 1)[-1], {})
    const = node.get("properties", {}).get("type", {}).get("const")
    return const if isinstance(const, str) else None


def _string_field(node: dict[str, Any]) -> dict[str, Any]:
    """A single-``string`` field carrying over ``node``'s title/description/default.

    A ``list`` default (a bare uniform vector such as ``[0, 0, 0]``) is rendered as
    the OpenFOAM ``(0 0 0)`` token so the collapsed text field shows a sane value.
    """
    out: dict[str, Any] = {"type": "string"}
    for k in ("title", "description"):
        if k in node:
            out[k] = node[k]
    default = node.get("default")
    if isinstance(default, str):
        out["default"] = default
    elif isinstance(default, list):
        out["default"] = "(" + " ".join(str(x) for x in default) + ")"
    return out


def _dimensions_field(node: dict[str, Any]) -> dict[str, Any]:
    """Render the fixed 7-element OpenFOAM dimension vector as one bracketed text field.

    JSONForms renders an ``array`` of integers as a growable list with a spinner and
    a delete button per element — nonsense for a fixed units vector. The field model
    round-trips the bracket token (``[0 1 -1 0 0 0 0]``), so a single string input is
    both cleaner and correct.
    """
    default = node.get("default")
    if isinstance(default, list):
        default = "[" + " ".join(str(int(x)) for x in default) + "]"
    out: dict[str, Any] = {
        "type": "string",
        "title": node.get("title", "Dimensions"),
        "description": "OpenFOAM unit exponents [kg m s K mol A cd]",
    }
    if isinstance(default, str):
        out["default"] = default
    return out


def _is_value_union(arms: list[Any]) -> bool:
    """A ``FieldValue``-style union: a bare ``string`` arm plus a numeric/array arm.

    These (``internalField``, a BC ``value`` / ``inletValue``) accept an OpenFOAM
    uniform literal, so they collapse to one text field. The BC *type* union is not
    matched — its arms are all ``$ref`` objects, none a bare ``string``/number.
    """
    has_str = any(isinstance(a, dict) and a.get("type") == "string" for a in arms)
    has_num = any(
        isinstance(a, dict) and a.get("type") in ("number", "integer", "array")
        for a in arms
    )
    return has_str and has_num


def jsonforms_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a pydantic JSON Schema render cleanly in JSONForms (generic, no per-config).

    Schema-driven rewrites, applied recursively (schema, ``$defs``, ``properties``,
    ``items``, ``additionalProperties`` and union arms):

    * **Unwrap ``Optional[X]``** — ``anyOf``/``oneOf`` with exactly one non-``null`` arm
      collapses to that arm (JSONForms otherwise renders the ``null`` arm as a second,
      empty ``"ANYOF"`` block — e.g. turbulence ``RAS``/``LES``).
    * **Collapse value unions** — a ``FieldValue`` union (uniform literal string +
      scalar/vector) becomes one ``string`` field instead of JSONForms'
      ``ANYOF-0``/``ANYOF-1``/``NONUNIFORM`` combinator tabs.
    * **Bracket the dimension vector** — a ``dimensions`` integer array becomes one
      bracketed ``string`` field, not a growable spinner list.
    * **Titled discriminated ``oneOf``** — a union whose arms carry a ``const`` ``type``
      discriminator (BC / scheme types) becomes a ``oneOf`` whose arms are titled by
      their ``const`` value, so JSONForms shows a clean "pick a type" dropdown
      (``fixedValue``/``noSlip``/…) instead of stacking every arm.
    """
    defs: dict[str, Any] = schema.get("$defs", {})

    def transform(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        node = dict(node)
        if isinstance(node.get("description"), str):
            node["description"] = _clean_description(node["description"])

        for comb in ("anyOf", "oneOf"):
            arms = node.get(comb)
            if not isinstance(arms, list):
                continue
            non_null = [
                a for a in arms if not (isinstance(a, dict) and a.get("type") == "null")
            ]
            if len(non_null) == 1 and len(non_null) < len(arms):
                merged = dict(non_null[0])
                for k, v in node.items():
                    if k != comb:
                        merged.setdefault(k, v)
                return transform(merged)
            if _is_value_union(non_null):
                return _string_field(node)
            consts = [_const_type_of(a, defs) for a in non_null if isinstance(a, dict)]
            if sum(c is not None for c in consts) >= 2:
                node.pop(comb, None)
                node["oneOf"] = [
                    ({**a, "title": c} if c else transform(a))
                    for a, c in zip(non_null, consts)
                ]
                break

        if isinstance(node.get("properties"), dict):
            node["properties"] = {
                k: (
                    _dimensions_field(v)
                    if k == "dimensions"
                    and isinstance(v, dict)
                    and v.get("type") == "array"
                    else _label_title(k, transform(v))
                )
                for k, v in node["properties"].items()
            }
        if isinstance(node.get("$defs"), dict):
            node["$defs"] = {k: transform(v) for k, v in node["$defs"].items()}
        if isinstance(node.get("items"), dict):
            node["items"] = transform(node["items"])
        if isinstance(node.get("additionalProperties"), dict):
            node["additionalProperties"] = transform(node["additionalProperties"])
        if isinstance(node.get("oneOf"), list):
            node["oneOf"] = [
                a if ("$ref" in a and "title" in a) else transform(a)
                for a in node["oneOf"]
            ]
        # A def that *is* a discriminated arm: title it by its const type.
        const = node.get("properties", {}).get("type", {}).get("const")
        if const and "title" in node:
            node["title"] = const
        return node

    result: dict[str, Any] = transform(schema)
    return result


def _dict_title(cls_name: str, file: str | None) -> str:
    """Panel title for a dict config: the OpenFOAM file it writes, qualified.

    The file basename is the domain language (``controlDict``, ``fvSchemes``); a
    config that co-owns a file with others gets its model/aspect as a qualifier —
    ``fvSchemes · Pimple``, ``controlDict · Courant``. Falls back to the humanized
    class name when the config is not file-bound.
    """
    stem = cls_name.removesuffix("Config")
    if "_" in stem:  # model-prefixed duplicates, e.g. "Pimple_fvSchemes"
        model, qualified = stem.split("_", 1)
        return f"{qualified} · {model}"
    if not file:
        return humanize(stem)
    base = file.rsplit("/", 1)[-1]
    if stem.lower() == base.lower():
        return base
    return f"{base} · {humanize(stem)}"  # co-owner: "controlDict · Courant"


def build_forms(solver: Any) -> list[FormEntry]:
    """All :class:`FormEntry` descriptors for ``solver``.

    Dict configs → one entry (step ``schemes`` for ``system/fv*`` configs, else
    ``models``). Field configs (``0/<name>``) → two entries: ``field_in`` (step
    ``initial``) and ``field_bc`` (step ``bcs``).
    """
    cfgs = configurations(solver)
    owner = _owner_by_cls_name(solver)
    entries: list[FormEntry] = []

    for info in tools.list_configs(solver):
        if info.file and info.file.rsplit("/", 1)[-1] in _MESH_FILES:
            continue  # meshing/preprocessing config — out of scope for the wizard
        dto = tools.config_schema(solver, info.name)
        cls = cfgs[info.cls_name]
        owner_model = owner.get(info.cls_name)

        if info.file and info.file.startswith("0/"):
            fname = field_name(cls)
            entries.append(
                FormEntry(
                    key=f"field_in:{info.cls_name}",
                    config_name=info.name,
                    cls_name=info.cls_name,
                    title=f"{fname} — initial value",
                    schema=jsonforms_schema(slice_schema(dto.json_schema, INPUT_KEYS)),
                    defaults=_slice_defaults(dto.defaults, INPUT_KEYS),
                    state_key=f"form_{info.name}__in",
                    kind="field_in",
                    step="initial",
                    owner_model=owner_model,
                    cls=cls,
                )
            )
            entries.append(
                FormEntry(
                    key=f"field_bc:{info.cls_name}",
                    config_name=info.name,
                    cls_name=info.cls_name,
                    title=f"{fname} — boundary conditions",
                    schema=jsonforms_schema(slice_schema(dto.json_schema, _BC_KEYS)),
                    defaults=_slice_defaults(dto.defaults, _BC_KEYS),
                    state_key=f"form_{info.name}__bc",
                    kind="field_bc",
                    step="bcs",
                    owner_model=owner_model,
                    cls=cls,
                )
            )
        else:
            step = "schemes" if is_scheme_config(cls) else "models"
            entries.append(
                FormEntry(
                    key=f"dict:{info.cls_name}",
                    config_name=info.name,
                    cls_name=info.cls_name,
                    title=_dict_title(info.cls_name, info.file),
                    schema=jsonforms_schema(dto.json_schema),
                    defaults=dto.defaults,
                    state_key=f"form_{info.name}",
                    kind="dict",
                    step=step,
                    owner_model=owner_model,
                    cls=cls,
                )
            )
    return entries


# Patch role → preferred BC type, most-preferred first. Clamped to the field's
# ``allowed_bc_types`` (a scalar field has no ``noSlip``), then to the first allowed
# non-fallback arm. Payload-free types are preferred so the scaffold stays valid;
# value-carrying arms (``fixedValue``) surface a ``value`` field for the user to fill.
# Role ``empty`` seeds a ``symmetry`` BC: the snappy mesh realises an empty (2-D)
# patch as a generalized ``symmetry`` patch (see ``ui.geometry._PATCH_TYPE``), and the
# BC type must match the mesh patch type.
_ROLE_BC_PREFS: dict[str, tuple[str, ...]] = {
    "empty": ("symmetry", "symmetryPlane", "empty"),
    "symmetry": ("symmetry", "symmetryPlane"),
    "wall": ("noSlip", "zeroGradient", "fixedValue"),
    "inlet": ("fixedValue", "zeroGradient"),
    "outlet": ("zeroGradient", "inletOutlet", "fixedValue"),
}


def allowed_bc_types(bc_entry: FormEntry) -> list[str]:
    """The BC ``type`` names a ``field_bc`` form allows (its ``boundaryField`` arms).

    Reads the titled ``oneOf`` arms produced by :func:`jsonforms_schema`; the untitled
    open-set fallback (``GenericBC``) has no ``title`` and is skipped.
    """
    ap = (
        bc_entry.schema.get("properties", {})
        .get("boundaryField", {})
        .get("additionalProperties", {})
    )
    arms = ap.get("oneOf") or ap.get("anyOf") or []
    out: list[str] = []
    for arm in arms:
        if isinstance(arm, dict) and isinstance(arm.get("title"), str):
            out.append(arm["title"])
    return out


def _bc_type_for_role(role: str, allowed: list[str]) -> str:
    """The best allowed BC type for a patch ``role`` (see :data:`_ROLE_BC_PREFS`)."""
    for pref in _ROLE_BC_PREFS.get(role, ()):
        if pref in allowed:
            return pref
    return allowed[0] if allowed else "zeroGradient"


# BC types with vector-only arms — their presence marks the field as a vector, so the
# seeded zero literal is ``uniform (0 0 0)`` rather than ``uniform 0``.
_VECTOR_MARKERS = ("noSlip", "pressureInletOutletVelocity")


def _bc_seed(type_name: str, is_vector: bool) -> dict[str, Any]:
    """A minimal *valid* BC payload for ``type_name``.

    ``fixedValue`` / ``inletOutlet`` require a ``value`` — seeding just ``{type}`` would
    fail their schema arm, so JSONForms' ``oneOf`` scorer falls back to the open
    ``GenericBC`` arm and renders a raw type box. Seeding a zero literal keeps the arm
    match (and the case valid) until the user sets the real value.
    """
    zero = "uniform (0 0 0)" if is_vector else "uniform 0"
    if type_name == "fixedValue":
        return {"type": type_name, "value": zero}
    if type_name == "inletOutlet":
        return {"type": type_name, "inletValue": zero, "value": zero}
    return {"type": type_name}


def seed_boundary_field(
    bc_entry: FormEntry, patches: list[dict[str, Any]], current: dict[str, Any]
) -> dict[str, Any]:
    """Merge discovered patches into a ``field_bc`` form's live data.

    Each scanned patch (``{"name", "role", ...}``) that isn't already present gets a
    ``{type: <role default>}`` entry, so the boundary-conditions step shows the real
    patch names with a sensible BC pre-selected instead of a blank property map.
    Existing (user- or AI-filled) patch entries are preserved.
    """
    allowed = allowed_bc_types(bc_entry)
    is_vector = any(m in allowed for m in _VECTOR_MARKERS)
    data = dict(current or {})
    bf = dict(data.get("boundaryField") or {})
    for patch in patches:
        name = patch.get("name")
        if name and name not in bf:
            bc_type = _bc_type_for_role(patch.get("role", ""), allowed)
            bf[name] = _bc_seed(bc_type, is_vector)
    data["boundaryField"] = bf
    return data


def patch_bc_schema(bc_entry: FormEntry, names: list[str]) -> dict[str, Any]:
    """Rebuild a ``field_bc`` schema with one concrete ``boundaryField`` property per patch.

    The default schema types ``boundaryField`` as an open ``additionalProperties`` map,
    which JSONForms renders as a bare "property name" editor with a generic type picker.
    Once the patch names are known (from the geometry scan) we pin them as titled object
    properties instead, so each patch renders as its own section with a clean BC-type
    dropdown. Returns ``bc_entry.schema`` unchanged when there are no names or no BC arms.
    """
    base = bc_entry.schema
    bf = base.get("properties", {}).get("boundaryField", {})
    value_schema = bf.get("additionalProperties")
    if not names or not isinstance(value_schema, dict):
        return base
    props = {name: {**value_schema, "title": name} for name in names}
    new_bf = {
        "type": "object",
        "title": bf.get("title", "Boundary field"),
        "properties": props,
    }
    return {
        **base,
        "properties": {**base.get("properties", {}), "boundaryField": new_bf},
    }
