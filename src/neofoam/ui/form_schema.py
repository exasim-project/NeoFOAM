# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Rewrite a pydantic JSON Schema so JSONForms renders it cleanly (headless, pure)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from neofoam.framework.validation import SOLVER_COMPANION
from neofoam.ui.boundary_forms import _arm_titles, _role_bc_seed

__all__ = [
    "ADDER_TRANSLATIONS",
    "RENDERER_KEYWORDS",
    "alternatives_uischema",
    "humanize",
    "inline_refs",
    "jsonforms_schema",
]

# Schema keywords the bundled JSONForms renderers (jsonforms_module/*.mjs) match by
# name; a rename on one side only falls back to the stock renderer without an error.
# Each: value shape · emitting function · renderer/layout.
_NF_COMPACT = "nfCompact"  # True · _tag_adders · NfCompactSection/rows
_NF_SOLVERS = "nfSolvers"  # True · _pin_solver_controls · NfCompactSection/solvers
_NF_SOLVER = "nfSolver"  # {solver: companion key} · _pin_solver_controls · NfCompactSection/grid
_NF_GRID = "nfGrid"  # True · _tag_scalar_grid · NfGridLayout
_NF_DICT = "nfDict"  # True · _dictionary_cards · NfCompactSection/grid
_NF_DICTS = "nfDicts"  # True · _dictionary_cards · NfCompactSection/solvers
_NF_PATCHES = "nfPatches"  # True · _patch_adder · NfCompactSection/rows
RENDERER_KEYWORDS = (
    _NF_COMPACT,
    _NF_SOLVERS,
    _NF_SOLVER,
    _NF_GRID,
    _NF_DICT,
    _NF_DICTS,
    _NF_PATCHES,
)

# The ``FoamFile`` header (version / format / class / object) a config may mirror is
# boilerplate for the writer: it stays in the form data but gets no form controls.
_FILE_HEADER = "FoamFile"


# camelCase / snake_case boundary for human labels: "writeControl" → "Write control".
# "NeoN" is a name, not two words: no break inside it, but one after it ("NeoN Control").
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])(?!(?<=Neo)N(?![a-z]))|(?<=NeoN)(?=[A-Z])")
# Sphinx roles in pydantic docstrings: ":class:`~a.b.C`" → "C".
_SPHINX_ROLE_RE = re.compile(r":[a-zA-Z:]+:`~?([^`]+)`")


def humanize(key: str) -> str:
    """A property key as a human label: ``deltaT`` → "Delta T", ``nu`` → "Nu"."""
    words = _CAMEL_RE.sub(" ", key.replace("_", " ")).strip()
    return words[:1].upper() + words[1:]


def _is_openfoam_block(node: Any) -> bool:
    """True when ``node`` models a raw OpenFOAM sub-dictionary rather than an API model.

    ``fv_configs._rebuild_sections`` synthesises one pydantic model per
    ``fvSchemes``/``fvSolution`` section (``create_model("_" + section, …)``), and
    ``_GravityHeader`` mirrors a ``FoamFile`` header the same way. The leading
    underscore in the class name — which pydantic copies into ``title`` — is the only
    trace of that distinction left in the JSON schema, and it is what tells the two
    kinds of key apart: a *block*'s property keys are the OpenFOAM entries themselves,
    written by the case author (``p_rghFinal``, ``div(phi,U)``, ``alpha.water``), while
    every other model is hand-written and its field names are an API surface
    (``writeControl``, ``momentumPredictor``) that reads naturally as prose.
    """
    return isinstance(node, dict) and str(node.get("title", "")).startswith("_")


def _label_title(
    key: str, node: Any, *, in_block: bool, class_names: frozenset[str] = frozenset()
) -> Any:
    """Title one property: verbatim for OpenFOAM keys, :func:`humanize`d for prose.

    ``in_block`` says the *owning* object is an OpenFOAM block, so ``key`` is a case
    entry; a node that is itself a block gets its key too (otherwise the synthesised
    ``_ddtSchemes`` class name leaks into the heading). An OpenFOAM key has no prose
    form — humanising it destroys information a user cannot recover (``p_rghFinal`` →
    "P Rghfinal"), and the key is what the written case file contains, so it is shown
    exactly as typed. Otherwise a pydantic auto-title (``"Writecontrol"``) is replaced
    by :func:`humanize`, and a hand-written title is kept. A nested model's title is its
    class name (``class_names``), inherited through the ``$ref`` — two ``PhaseTransport``
    cards tell the user nothing — so it counts as auto as well.
    """
    if not isinstance(node, dict):
        return node
    if in_block or _is_openfoam_block(node):
        return {**node, "title": key}
    title = node.get("title")
    auto = (
        title is None
        or title in class_names
        or (isinstance(title, str) and title.replace(" ", "").lower() == key.lower())
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


def _hidden_discriminator(const: str) -> dict[str, Any]:
    """A ``type`` discriminator that stays in the data but renders no control.

    The union's arm selector already names the type, so a second "Type" dropdown with
    that one value only doubles the form's height. JSONForms generates a control per
    property it can derive a JSON type for; a bare ``const`` gives it none, so the
    property is skipped. ``default`` is what puts the value into the data JSONForms
    creates when the user switches arm, and ``const`` still validates it.
    """
    return {"const": const, "default": const}


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
        isinstance(a, dict) and a.get("type") in ("number", "integer", "array") for a in arms
    )
    return has_str and has_num


def _accepts_new_keys(node: dict[str, Any]) -> bool:
    """Whether the "add a key" row of an ``additionalProperties: true`` object is useful.

    JSONForms appends a "Property Name [+]" row to every such object. It earns its
    place on a collection: an OpenFOAM section whose entries all share one shape
    (``divSchemes`` gains a ``div(phi,k)``, ``solvers`` a solver block), or an object
    declaring no keys at all (a solver block, ``fvOptions``), which is edited through
    that row alone. A card of differently-shaped entries (``PIMPLE``, a phase's
    transport, a whole ``fvSolution``) is fixed; there the keyword is dropped, which
    hides the row and — absent meaning "allowed" — still validates the same data.
    """
    shapes = [
        {k: v for k, v in prop.items() if k not in ("title", "default")}
        for prop in node.get("properties", {}).values()
    ]
    if not shapes:
        return True
    return _is_openfoam_block(node) and all(shape == shapes[0] for shape in shapes)


# What each kind of "add a key" row adds, keyed by the ``i18n`` prefix put on its schema
# node: the name field's label (JSONForms would say "Property Name" for all of them) and
# the button's text, in the order of `_ADDER_SUFFIXES`.
_ADDER_SUFFIXES = ("propertyNameLabel", "addLabel")
_ADDER_LABELS = {
    "nf.patch": ("Patch name, e.g. inlet", "Add patch"),
    "nf.entry": ("Entry, e.g. div(phi,k)", "Add entry"),
    "nf.solver": ("Field, e.g. p", "Add solver"),
    "nf.option": ("Option, e.g. maxIter", "Add option"),
    "nf.zone": ("Zone name, e.g. MRF1", "Add zone"),
    "nf.source": ("Source name, e.g. momentumSource", "Add source"),
    "nf.keyword": ("Keyword, e.g. cellZone", "Add keyword"),
}

ADDER_TRANSLATIONS: dict[str, str] = {
    f"{prefix}.{suffix}": text
    for prefix, texts in _ADDER_LABELS.items()
    for suffix, text in zip(_ADDER_SUFFIXES, texts, strict=True)
}
"""JSONForms translations for the "add a key" rows (the ``<json-forms>`` ``translations``)."""


def _is_open_dict(node: Any) -> bool:
    """An object declaring no keys at all (a solver block): edited through its adder alone."""
    return (
        isinstance(node, dict)
        and node.get("additionalProperties") is True
        and not node.get("properties")
    )


def _tag_adders(node: dict[str, Any]) -> None:
    """Mark an OpenFOAM section's "add a key" rows with what they add (mutates ``node``).

    A section of scheme unions additionally gets ``nfCompact``: the bundled renderer
    then draws it as one row per entry instead of JSONForms' stacked full-width selects.
    """
    if not _is_openfoam_block(node):
        return
    shapes = list(node.get("properties", {}).values())
    for shape in shapes:
        if _is_open_dict(shape):
            shape["i18n"] = "nf.option"
    if node.get("additionalProperties") is not True or not shapes:
        return
    if all("oneOf" in shape for shape in shapes):
        node["i18n"] = "nf.entry"
        node[_NF_COMPACT] = True
    else:
        node["i18n"] = "nf.solver"


_PRECONDITIONERS = ["DIC", "FDIC", "DILU", "diagonal", "GAMG", "none"]
_SMOOTHERS = ["symGaussSeidel", "GaussSeidel", "DICGaussSeidel", "DIC", "DILU"]


# The keys every linear-solver block has. Choices are ``examples``, never an ``enum``: a
# loaded case may name a solver OpenFOAM does not know (``Ginkgo``), and a
# GAMG-preconditioned PCG nests a dictionary under ``preconditioner``.
_SOLVER_CONTROLS: dict[str, dict[str, Any]] = {
    "solver": {"type": "string", "examples": list(SOLVER_COMPANION)},
    "preconditioner": {"type": ["string", "object"], "examples": _PRECONDITIONERS},
    "smoother": {"type": "string", "examples": _SMOOTHERS},
    "tolerance": {"type": "number"},
    "relTol": {"type": "number"},
}


def _companion_rules() -> list[dict[str, Any]]:
    """One ``if solver … then required`` rule per companion key, so an empty one shows red."""
    rules = []
    for companion in dict.fromkeys(SOLVER_COMPANION.values()):
        solvers = [name for name, key in SOLVER_COMPANION.items() if key == companion]
        selected = {"properties": {"solver": {"enum": solvers}}, "required": ["solver"]}
        rules.append({"if": selected, "then": {"required": [companion]}})
    return rules


def _pin_solver_controls(node: dict[str, Any]) -> None:
    """Draw the ``solvers`` section's blocks as grids of their standard keys (mutates ``node``).

    The config publishes a block as an open dictionary (any key, written back as read);
    only the form declares the keys, so the bundled renderer can offer dropdowns and
    number fields.
    """
    if node.get("title") != "_solvers":
        return
    node[_NF_SOLVERS] = True
    for block in node.get("properties", {}).values():
        block[_NF_SOLVER] = SOLVER_COMPANION
        block["allOf"] = _companion_rules()
        block["properties"] = {
            key: {**control, "title": key} for key, control in _SOLVER_CONTROLS.items()
        }


_SCALAR_TYPES = ("number", "integer", "boolean")


def _is_scalar(node: Any) -> bool:
    """A property drawn as one small control: a number, a checkbox or a fixed choice."""
    return isinstance(node, dict) and (node.get("type") in _SCALAR_TYPES or "enum" in node)


def _tag_scalar_grid(node: dict[str, Any]) -> None:
    """Flag an object of two or more scalars only with ``nfGrid`` (mutates ``node``).

    The bundled layout renderer then sets its controls side by side in a wrapping grid
    instead of one full-width control per line (``PIMPLE``, a model's coefficients).
    """
    shapes = list(node.get("properties", {}).values())
    if len(shapes) >= 2 and all(_is_scalar(shape) for shape in shapes):
        node[_NF_GRID] = True


# A config that is one open dictionary of named sub-dictionaries → what a name is there.
_DICTIONARY_CONFIGS = {"MRFPropertiesConfig": "nf.zone", "FvOptionsConfig": "nf.source"}


def _dictionary_cards(schema: dict[str, Any]) -> dict[str, Any]:
    """A :data:`_DICTIONARY_CONFIGS` schema as one card per sub-dictionary, each a keyword grid.

    Such a config declares no property, so JSONForms' generated layout is empty and the
    panel would show nothing; the bundled section renderer draws ``nfDicts``/``nfDict``.
    """
    prefix = _DICTIONARY_CONFIGS.get(schema.get("title", ""))
    if prefix is None:
        return schema
    card = {"type": "object", "additionalProperties": True, _NF_DICT: True, "i18n": "nf.keyword"}
    # The panel already names the file; the class name would head the section a second time.
    untitled = {key: value for key, value in schema.items() if key != "title"}
    return {**untitled, "additionalProperties": card, _NF_DICTS: True, "i18n": prefix}


def _patch_adder(node: dict[str, Any]) -> None:
    """Draw a ``boundaryField`` map as patch rows whose adder adds a patch (mutates ``node``).

    Keyed on the property name, so no caller can forget it: the stock adder corrupts a
    dotted patch name. ``nfPatches`` picks the bundled row renderer, which binds a patch
    through the map's data, so a name holding ``.`` (``wall.left``) works. The row is
    labelled as a patch-name box, and the BC union is typed ``object``: a new key is
    seeded from the ``type``, and with none it would be the string ``""``. Its
    ``default`` is the BC a scan seeds on a wall, so a hand-added patch shows and saves
    a real type from the start.
    """
    boundary_field = node.get("properties", {}).get("boundaryField")
    union = boundary_field.get("additionalProperties") if boundary_field else None
    if not isinstance(union, dict):
        return
    boundary_field = {**boundary_field, "i18n": "nf.patch", _NF_PATCHES: True}
    boundary_field["additionalProperties"] = {
        **union,
        "type": "object",
        "default": _role_bc_seed("wall", _arm_titles(union)),
    }
    node["properties"]["boundaryField"] = boundary_field


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve every ``#/$defs/...`` ``$ref`` into a self-contained subtree.

    JSONForms' combinator renderers hand each ``oneOf``/``anyOf`` arm to AJV
    *standalone* (``ajv.compile(subschema)``) to find the fitting arm. An arm
    that is (or contains) a ``$ref`` cannot be compiled that way — AJV throws,
    the result is never cached, and the schema is recompiled on every reactive
    re-evaluation. For the big scheme configs (deeply nested discriminated
    unions) that compile-throw loop hard-freezes the page. Inlining makes every
    subschema self-contained, so AJV compiles each arm once and caches it.

    Sibling keys of a ``$ref`` (e.g. the arm ``title``) override the resolved
    definition. A union arm that recurses into a definition being resolved is
    dropped, which bounds a self-referential union at one level: inside
    ``cellLimited`` the inner gradient scheme offers the non-recursive arms only
    (OpenFOAM cases do not nest a limiter in a limiter). Other cyclic refs, and
    unknown ones, are left in place; ``discriminator`` keys are dropped
    (JSONForms ignores them, and their ``mapping`` would dangle once the defs are
    gone). ``$defs`` is removed when nothing refers to it anymore.
    """
    defs = schema.get("$defs", {})

    def recurses(arm: Any, stack: frozenset[str]) -> bool:
        ref = arm.get("$ref") if isinstance(arm, dict) else None
        return isinstance(ref, str) and ref.rsplit("/", 1)[-1] in stack

    def resolve(node: Any, stack: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, stack) for item in node if not recurses(item, stack)]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.rsplit("/", 1)[-1]
            if name not in defs or name in stack:
                return dict(node)  # unknown or cyclic: keep the $ref
            target = resolve(defs[name], stack | {name})
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            return {**target, **{k: resolve(v, stack) for k, v in siblings.items()}}
        return {
            k: resolve(v, stack) for k, v in node.items() if k not in ("$defs", "discriminator")
        }

    def has_ref(node: Any) -> bool:
        if isinstance(node, list):
            return any(has_ref(item) for item in node)
        if isinstance(node, dict):
            return "$ref" in node or any(has_ref(v) for v in node.values())
        return False

    out: dict[str, Any] = resolve(schema, frozenset())
    if has_ref(out):
        out["$defs"] = defs  # a cyclic ref survived — keep its definitions
    return out


@dataclass(frozen=True)
class _SchemaNames:
    """What rewriting one node needs from the whole schema."""

    defs: dict[str, Any]
    """The ``$defs`` :func:`inline_refs` had to keep (cyclic refs)."""
    class_names: frozenset[str]
    """Every model class name: a title equal to one is an auto-title."""


def _unwrapped_optional(node: dict[str, Any], comb: str, arm: dict[str, Any]) -> dict[str, Any]:
    """``Optional[X]`` as ``X``: the one non-null ``arm`` plus the keys only ``node`` has."""
    merged = dict(arm)
    for k, v in node.items():
        if k != comb:
            merged.setdefault(k, v)
    return merged


def _title_arms(node: dict[str, Any], comb: str, non_null: list[Any], names: _SchemaNames) -> bool:
    """Make a discriminated union a ``oneOf`` titled by the arms' ``const`` (mutates ``node``)."""
    consts = [_const_type_of(a, names.defs) for a in non_null if isinstance(a, dict)]
    if sum(c is not None for c in consts) < 2:
        return False
    node.pop(comb, None)
    node["oneOf"] = [
        ({**a, "title": c} if c else _transform(a, names)) for a, c in zip(non_null, consts)
    ]
    return True


def _non_null_arms(node: dict[str, Any], comb: str) -> list[Any]:
    """The arms of ``node``'s ``comb`` union that are not ``null``; none without that union."""
    arms = node.get(comb)
    if not isinstance(arms, list):
        return []
    return [a for a in arms if not (isinstance(a, dict) and a.get("type") == "null")]


def _rewrite_union(node: dict[str, Any], names: _SchemaNames) -> dict[str, Any] | None:
    """The node a collapsing union becomes, else ``None`` (its arms then titled in ``node``)."""
    for comb in ("anyOf", "oneOf"):
        non_null = _non_null_arms(node, comb)
        if len(non_null) == 1 and len(non_null) < len(node[comb]):
            return _transform_object(_unwrapped_optional(node, comb, non_null[0]), names)
        if _is_value_union(non_null):
            return _string_field(node)
        if _title_arms(node, comb, non_null, names):
            break
    return None


def _transform_property(key: str, value: Any, in_block: bool, names: _SchemaNames) -> Any:
    """One rewritten, titled property; ``dimensions`` is keyed on its name."""
    if key == "dimensions" and isinstance(value, dict) and value.get("type") == "array":
        return _dimensions_field(value)
    return _label_title(
        key, _transform(value, names), in_block=in_block, class_names=names.class_names
    )


def _transform_properties(node: dict[str, Any], names: _SchemaNames) -> None:
    """Rewrite and title ``node``'s properties, minus the file header (mutates ``node``)."""
    if not isinstance(node.get("properties"), dict):
        return
    in_block = _is_openfoam_block(node)
    node["properties"] = {
        k: _transform_property(k, v, in_block, names)
        for k, v in node["properties"].items()
        if k != _FILE_HEADER
    }


def _transform_arm(arm: Any, names: _SchemaNames) -> Any:
    """One rewritten ``oneOf`` arm; a titled ``$ref`` (a cycle) stays as it is."""
    return arm if ("$ref" in arm and "title" in arm) else _transform(arm, names)


def _transform_children(node: dict[str, Any], names: _SchemaNames) -> None:
    """Rewrite every sub-schema of ``node`` (mutates ``node``)."""
    _transform_properties(node, names)
    if isinstance(node.get("$defs"), dict):
        node["$defs"] = {k: _transform(v, names) for k, v in node["$defs"].items()}
    for key in ("items", "additionalProperties"):
        if isinstance(node.get(key), dict):
            node[key] = _transform(node[key], names)
    if isinstance(node.get("oneOf"), list):
        node["oneOf"] = [_transform_arm(a, names) for a in node["oneOf"]]


def _drop_useless_adder(node: dict[str, Any]) -> None:
    """Hide the "add a key" row where :func:`_accepts_new_keys` says no (mutates ``node``)."""
    if node.get("additionalProperties") is True and not _accepts_new_keys(node):
        del node["additionalProperties"]


def _hide_discriminator(node: dict[str, Any]) -> None:
    """Title a discriminated arm by its ``const`` type and hide that property (mutates ``node``)."""
    const = node.get("properties", {}).get("type", {}).get("const")
    if const and "title" in node:
        node["title"] = const
    if const:
        node["properties"]["type"] = _hidden_discriminator(const)


# The passes run on every node, in this order, once its children are rewritten, so a
# pass sees finished children. Two ordering constraints; the other passes commute:
# * _drop_useless_adder before _tag_adders: a label goes on a row that is still there
#   (_tag_adders reads ``additionalProperties``); a fixed card such as PIMPLE gets none.
# * _tag_adders before _pin_solver_controls: a solver block is recognised as an open
#   dict (``nf.option``) by having no ``properties``, which pinning then gives it.
_NODE_PASSES = (
    _drop_useless_adder,
    _tag_adders,
    _pin_solver_controls,
    _patch_adder,
    _tag_scalar_grid,
    _hide_discriminator,
)


def _transform(node: Any, names: _SchemaNames) -> Any:
    """``node`` rewritten for JSONForms when it is a schema object, else as it is."""
    return _transform_object(node, names) if isinstance(node, dict) else node


def _transform_object(node: dict[str, Any], names: _SchemaNames) -> dict[str, Any]:
    """One schema object rewritten: union, then children, then :data:`_NODE_PASSES`."""
    node = dict(node)
    if isinstance(node.get("description"), str):
        node["description"] = _clean_description(node["description"])
    collapsed = _rewrite_union(node, names)
    if collapsed is not None:
        return collapsed
    _transform_children(node, names)
    for node_pass in _NODE_PASSES:
        node_pass(node)
    return node


def jsonforms_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a pydantic JSON Schema render cleanly in JSONForms (generic, no per-config).

    The schema is first made self-contained via :func:`inline_refs` (JSONForms'
    combinator renderers cannot handle ``$ref`` arms — see there). Then the
    schema-driven rewrites, applied recursively (schema, ``$defs``,
    ``properties``, ``items``, ``additionalProperties`` and union arms):

    * **Unwrap ``Optional[X]``** — ``anyOf``/``oneOf`` with exactly one non-``null`` arm
      collapses to that arm (JSONForms otherwise renders the ``null`` arm as a second,
      empty ``"ANYOF"`` block — e.g. turbulence ``RAS``/``LES``).
    * **Collapse value unions** — a ``FieldValue`` union (uniform literal string +
      scalar/vector) becomes one ``string`` field instead of JSONForms'
      ``ANYOF-0``/``ANYOF-1``/``NONUNIFORM`` combinator tabs.
    * **Bracket the dimension vector** — a ``dimensions`` integer array becomes one
      bracketed ``string`` field, not a growable spinner list.
    * **Title properties** — a pydantic auto-title becomes a human label, except inside
      an OpenFOAM block (:func:`_is_openfoam_block`), whose keys are shown verbatim.
    * **Titled discriminated ``oneOf``** — a union whose arms carry a ``const`` ``type``
      discriminator (BC / scheme types) becomes a ``oneOf`` whose arms are titled by
      their ``const`` value, so JSONForms shows a clean "pick a type" dropdown
      (``fixedValue``/``noSlip``/…) instead of stacking every arm. The arm's own
      ``type`` property is kept in the data but not rendered (:func:`_hidden_discriminator`).
    * **No ``FoamFile`` header** — writer boilerplate; kept in the data, not rendered.
    * **Useful "add a key" rows only** — see :func:`_accepts_new_keys`.
    * **Labelled "add a key" rows, compact scheme sections** — see :func:`_tag_adders`.
    * **Linear-solver blocks as grids** — see :func:`_pin_solver_controls`.
    * **All-scalar objects as grids** — see :func:`_tag_scalar_grid`.
    * **Patch rows for ``boundaryField``** — see :func:`_patch_adder`.
    * **Open configs as cards** — see :func:`_dictionary_cards` (whole schema, last).
    """
    class_names = frozenset(schema.get("$defs", {}))
    schema = inline_refs(schema)
    names = _SchemaNames(defs=schema.get("$defs", {}), class_names=class_names)
    return _dictionary_cards(_transform(schema, names))


def alternatives_uischema(schema: dict[str, Any]) -> dict[str, Any] | None:
    """A UISchema hiding each alternative sub-block until its selector names it.

    ``turbulenceProperties`` is a choice, not a set: ``simulationType`` is an enum whose
    values name the sibling sub-dictionaries (``RAS``/``LES``), and the config's
    validator rejects any block the type does not name. JSONForms' generated layout has
    no notion of that, so it stacks *every* block — a ``laminar`` case renders a full
    RAS **and** LES block, each red for its required-but-empty model field.

    When a schema shows that shape — an enum property whose values name two or more
    sibling object properties — this returns the layout JSONForms would have generated
    (a ``VerticalLayout`` of one ``Control`` per property), with a SHOW ``rule`` on each
    named block so only the selected one renders. Any other schema returns ``None``,
    i.e. keep the auto-layout.
    """
    props = schema.get("properties")
    if not isinstance(props, dict):
        return None
    for selector, node in props.items():
        if not isinstance(node, dict) or not isinstance(node.get("enum"), list):
            continue
        blocks = [
            v
            for v in node["enum"]
            if isinstance(props.get(v), dict) and props[v].get("type") == "object"
        ]
        if len(blocks) < 2:
            continue
        elements: list[dict[str, Any]] = []
        for name in props:
            element: dict[str, Any] = {"type": "Control", "scope": f"#/properties/{name}"}
            if name in blocks:
                element["rule"] = {
                    "effect": "SHOW",
                    "condition": {
                        "scope": f"#/properties/{selector}",
                        "schema": {"const": name},
                    },
                }
            elements.append(element)
        return {"type": "VerticalLayout", "elements": elements}
    return None
