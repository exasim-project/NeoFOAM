# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Build the JSONForms form descriptors for a solver (headless, pure).

Each :class:`FormEntry` carries the JSON Schema + defaults that drive a client-side
``<json-forms>`` component, plus the trame ``state_key`` under which that form's live
data lives, the owning selectable model (for hide/skip) and the wizard step it belongs
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
    "ADDER_TRANSLATIONS",
    "FormEntry",
    "build_forms",
    "build_field_forms",
    "build_mesh_forms",
    "exclusive_model_families",
    "humanize",
    "inline_refs",
    "jsonforms_schema",
    "alternatives_uischema",
    "allowed_bc_types",
    "seed_boundary_field",
    "patch_bc_schema",
]

_BC_KEYS = ("boundaryField",)

# The ``FoamFile`` header (version / format / class / object) a config may mirror is
# boilerplate for the writer: it stays in the form data but gets no form controls.
_FILE_HEADER = "FoamFile"

# Mesh / preprocessing dict configs belong to the upstream meshing stage (the
# geometry→mesh workflow), not the physics case wizard. They also aren't writable via
# write_configs' merged path (preprocess.yaml uses YAMLStrategy). Exclude them.
_MESH_FILES = frozenset({"blockMeshDict", "snappyHexMeshDict", "preprocess.yaml"})

# The subset of mesh dicts that are per-config *sweepable*: OpenFOAM dict configs
# with a real ``@IOStrategy`` file binding (``write_configs`` writes them cleanly).
# ``preprocess.yaml`` is excluded — it uses ``YAMLStrategy`` and is not a solver
# config the sweep applies. These feed the Parameters step's keyed mesh dimension
# only (never the physics wizard); see :func:`build_mesh_forms`.
_SWEEPABLE_MESH_FILES = frozenset({"blockMeshDict", "snappyHexMeshDict"})


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
    """Selectable model that owns this config (hide/skip); ``None`` ⇒ always shown."""
    cls: Any = field(default=None, compare=False)
    """The config class (used by field-half merge in ``case_spec``)."""
    uischema: dict[str, Any] | None = None
    """Optional JSONForms UISchema (``None`` ⇒ auto-layout from the schema)."""


def exclusive_model_families(solver: Any) -> dict[str, list[str]]:
    """Required model families with more than one member → their member names.

    ``SolverSpec.models(family, required=True)`` means *exactly one* member runs per
    case (the family resolves it with ``detect_and_create()``), so such a family's
    members are alternatives — Pimple **or** Simple, one turbulence model — whereas an
    optional family's members are independent toggles (``detect_models()`` returns zero
    or more). That required flag plus the family's ``all_specs()`` is the whole signal;
    the UI needs no list of its own. A required family with a single member has nothing
    to choose and is omitted (its configs are always on). Keyed by the family class
    name; members keep registration order, so the first is the default choice.
    """
    families: dict[str, list[str]] = {}
    for family in solver.required_model_specs:
        members = [spec.name for spec in family.all_specs()]
        if len(members) > 1:
            families[family.__name__] = members
    return families


def _owner_by_cls_name(solver: Any) -> dict[str, str]:
    """Map each *gated*-model-owned config class name → its model name.

    A gated model is one the user selects: an optional model (a toggle) or a member of
    a mutually exclusive required family (:func:`exclusive_model_families`, a choice).
    Both are hidden — and skipped on save — while unselected. Configs owned only by an
    always-on required model (or by no model) are absent → ``None`` owner ⇒ never
    hidden.

    Two kinds of config stay ungated even though a gated model owns them, because they
    are common to the *whole* family rather than evidence for one member:

    * one owned by more than one gated model (``0/U`` and ``0/p`` are declared by both
      Pimple and Simple, ``turbulenceProperties`` by every turbulence model) — hiding
      it with either member would hide it from the case;
    * a ``0/<field>`` file of an exclusive family: the fields are the case's state,
      solved by whichever member is active, so only its *dictionaries* (the per-member
      ``fvSchemes``/``fvSolution`` slices) discriminate. Optional-model fields stay
      gated — those models *add* fields (Boussinesq's ``0/T``) rather than share them.
    """
    exclusive = {name for members in exclusive_model_families(solver).values() for name in members}
    owners: dict[str, list[str]] = {}
    for entry in tools.model_catalog(solver):
        if entry.required and entry.name not in exclusive:
            continue
        gated = entry.dicts if entry.name in exclusive else (*entry.dicts, *entry.fields)
        for cls_name in gated:
            owners.setdefault(cls_name, []).append(entry.name)
    return {cls_name: names[0] for cls_name, names in owners.items() if len(names) == 1}


def _slice_defaults(defaults: dict[str, Any], keep: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in defaults.items() if k in keep}


# camelCase / snake_case boundary for human labels: "writeControl" → "Write control".
# "NeoN" is a name, not two words: no break inside it, but one after it ("NeoN Control").
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])(?!(?<=Neo)N(?![a-z]))|(?<=NeoN)(?=[A-Z])")
# Sphinx roles in pydantic docstrings: ":class:`~a.b.C`" → "C".
_SPHINX_ROLE_RE = re.compile(r":[a-zA-Z:]+:`~?([^`]+)`")


def js_identifier(name: str) -> str:
    """``name`` as a JS identifier: trame state vars are evaluated as Vue expressions,

    so a field like ``alpha.water`` would otherwise read as ``alpha`` . ``water…``.
    """
    return re.sub(r"\W", "_", name)


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
# node. JSONForms would label all of them "Property Name".
_ADDER_LABELS = {
    "nf.patch": "Patch name, e.g. inlet",
    "nf.entry": "Entry, e.g. div(phi,k)",
    "nf.solver": "Field, e.g. p",
    "nf.option": "Option, e.g. maxIter",
}

# JSONForms addresses form data by dotted path (``boundaryField.inlet.type``), so a key
# holding one of these characters would be read and written at the wrong place.
_ADDER_NAME_INVALID = (
    "A name containing . [ or ] cannot be added here: the form library reads them as"
    " path separators."
)

ADDER_TRANSLATIONS: dict[str, str] = {
    f"{prefix}.{key}": text
    for prefix, label in _ADDER_LABELS.items()
    for key, text in (("propertyNameLabel", label), ("propertyNameInvalid", _ADDER_NAME_INVALID))
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
        node["nfCompact"] = True
    else:
        node["i18n"] = "nf.solver"


# Which key sits next to ``solver`` in a linear-solver block: a Krylov solver takes a
# preconditioner, a smoothing one a smoother. These are the OpenFOAM solvers the NeoN
# backend maps too, so every suggestion runs on both.
_SOLVER_COMPANION = {
    "PCG": "preconditioner",
    "PBiCGStab": "preconditioner",
    "PBiCG": "preconditioner",
    "smoothSolver": "smoother",
    "GAMG": "smoother",
}
_PRECONDITIONERS = ["DIC", "FDIC", "DILU", "diagonal", "GAMG", "none"]
_SMOOTHERS = ["symGaussSeidel", "GaussSeidel", "DICGaussSeidel", "DIC", "DILU"]


# The keys every linear-solver block has. Choices are ``examples``, never an ``enum``: a
# loaded case may name a solver OpenFOAM does not know (``Ginkgo``), and a
# GAMG-preconditioned PCG nests a dictionary under ``preconditioner``.
_SOLVER_CONTROLS: dict[str, dict[str, Any]] = {
    "solver": {"type": "string", "examples": list(_SOLVER_COMPANION)},
    "preconditioner": {"type": ["string", "object"], "examples": _PRECONDITIONERS},
    "smoother": {"type": "string", "examples": _SMOOTHERS},
    "tolerance": {"type": "number"},
    "relTol": {"type": "number"},
}


def _pin_solver_controls(node: dict[str, Any]) -> None:
    """Draw the ``solvers`` section's blocks as grids of their standard keys (mutates ``node``).

    The config keeps a block schemaless (any key, written back as read); only the form
    declares the keys, so the bundled renderer can offer dropdowns and number fields.
    """
    if node.get("title") != "_solvers":
        return
    node["nfSolvers"] = True
    for block in node.get("properties", {}).values():
        block["nfSolver"] = _SOLVER_COMPANION
        block["properties"] = {
            key: {**control, "title": key} for key, control in _SOLVER_CONTROLS.items()
        }


def _patch_adder(schema: dict[str, Any]) -> dict[str, Any]:
    """A sliced ``boundaryField`` schema whose "add a key" row adds a patch.

    The row is labelled as a patch-name box, and the BC union is typed ``object``:
    JSONForms seeds a new key from the ``type``, and with none it seeds the string ``""``.
    """
    props = schema["properties"]
    boundary_field = dict(props["boundaryField"], i18n="nf.patch")
    boundary_field["additionalProperties"] = {
        **boundary_field["additionalProperties"],
        "type": "object",
    }
    return {**schema, "properties": {**props, "boundaryField": boundary_field}}


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
    """
    class_names = frozenset(schema.get("$defs", {}))
    schema = inline_refs(schema)
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
            non_null = [a for a in arms if not (isinstance(a, dict) and a.get("type") == "null")]
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
                    ({**a, "title": c} if c else transform(a)) for a, c in zip(non_null, consts)
                ]
                break

        if isinstance(node.get("properties"), dict):
            in_block = _is_openfoam_block(node)
            node["properties"] = {
                k: (
                    _dimensions_field(v)
                    if k == "dimensions" and isinstance(v, dict) and v.get("type") == "array"
                    else _label_title(k, transform(v), in_block=in_block, class_names=class_names)
                )
                for k, v in node["properties"].items()
                if k != _FILE_HEADER
            }
        if isinstance(node.get("$defs"), dict):
            node["$defs"] = {k: transform(v) for k, v in node["$defs"].items()}
        if isinstance(node.get("items"), dict):
            node["items"] = transform(node["items"])
        if isinstance(node.get("additionalProperties"), dict):
            node["additionalProperties"] = transform(node["additionalProperties"])
        if isinstance(node.get("oneOf"), list):
            node["oneOf"] = [
                a if ("$ref" in a and "title" in a) else transform(a) for a in node["oneOf"]
            ]
        if node.get("additionalProperties") is True and not _accepts_new_keys(node):
            del node["additionalProperties"]
        _tag_adders(node)
        _pin_solver_controls(node)
        # A def that *is* a discriminated arm: title it by its const type.
        const = node.get("properties", {}).get("type", {}).get("const")
        if const and "title" in node:
            node["title"] = const
        if const:
            node["properties"]["type"] = _hidden_discriminator(const)
        return node

    result: dict[str, Any] = transform(schema)
    return result


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

    seen: set[str] = set()

    for info in tools.list_configs(solver):
        if info.file and info.file.rsplit("/", 1)[-1] in _MESH_FILES:
            continue  # meshing/preprocessing config — out of scope for the wizard
        # Mutually exclusive model families (incompressibleFluid binds both Pimple
        # and Simple) each declare the same field config. The classes are distinct
        # objects but schema-identical and bound to the same 0/<field> file, so the
        # wizard shows one form per config instead of two that overwrite each other.
        if info.name in seen:
            continue
        seen.add(info.name)
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
                    state_key=js_identifier(f"form_{info.name}__in"),
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
                    schema=_patch_adder(jsonforms_schema(slice_schema(dto.json_schema, _BC_KEYS))),
                    defaults=_slice_defaults(dto.defaults, _BC_KEYS),
                    state_key=js_identifier(f"form_{info.name}__bc"),
                    kind="field_bc",
                    step="bcs",
                    owner_model=owner_model,
                    cls=cls,
                )
            )
        else:
            step = "schemes" if is_scheme_config(cls) else "models"
            schema = jsonforms_schema(dto.json_schema)
            entries.append(
                FormEntry(
                    key=f"dict:{info.cls_name}",
                    config_name=info.name,
                    cls_name=info.cls_name,
                    title=_dict_title(info.cls_name, info.file),
                    schema=schema,
                    defaults=dto.defaults,
                    state_key=js_identifier(f"form_{info.name}"),
                    kind="dict",
                    step=step,
                    owner_model=owner_model,
                    cls=cls,
                    uischema=alternatives_uischema(schema),
                )
            )
    return entries


def build_mesh_forms(solver: Any) -> list[FormEntry]:
    """:class:`FormEntry` descriptors for the sweepable *mesh* dicts.

    These are the ``blockMeshDict`` / ``snappyHexMeshDict`` configs that
    :func:`build_forms` deliberately drops from the physics wizard
    (``_MESH_FILES``). The Parameters step surfaces them as sources of its keyed
    ``mesh`` dimension, so it needs their titles + JSONForms schemas; every entry
    is a plain ``dict``-kind ``FormEntry`` with ``step="mesh"`` (never a wizard
    step) and no owner model. Same shape as a wizard dict entry, so the sweep
    palette can treat them uniformly.
    """
    cfgs = configurations(solver)
    entries: list[FormEntry] = []
    for info in tools.list_configs(solver):
        base = info.file.rsplit("/", 1)[-1] if info.file else ""
        if base not in _SWEEPABLE_MESH_FILES:
            continue
        dto = tools.config_schema(solver, info.name)
        entries.append(
            FormEntry(
                key=f"dict:{info.cls_name}",
                config_name=info.name,
                cls_name=info.cls_name,
                title=_dict_title(info.cls_name, info.file),
                schema=jsonforms_schema(dto.json_schema),
                defaults=dto.defaults,
                state_key=js_identifier(f"form_{info.name}"),
                kind="dict",
                step="mesh",
                owner_model=None,
                cls=cfgs[info.cls_name],
            )
        )
    return entries


def build_field_forms(solver: Any) -> list[FormEntry]:
    """:class:`FormEntry` descriptors for the *whole* ``0/<field>`` configs.

    The wizard splits each field config into an *input* half (``field_in``) and a
    *boundary-conditions* half (``field_bc``) that render in separate steps. The
    Parameters step, though, sweeps a config as one unit (``apply_configs`` writes
    the whole ``0/<field>`` file), so it needs a single ``dict``-kind entry per
    field carrying the full schema (``dimensions`` + ``internalField`` +
    ``boundaryField``). That is what this builds — one whole-field entry per
    ``0/<name>`` config, ``step="fields"``, so a case author can sweep a boundary
    value (e.g. the inlet velocity) as its own dimension.
    """
    cfgs = configurations(solver)
    owner = _owner_by_cls_name(solver)
    entries: list[FormEntry] = []
    for info in tools.list_configs(solver):
        if not (info.file and info.file.startswith("0/")):
            continue
        dto = tools.config_schema(solver, info.name)
        fname = field_name(cfgs[info.cls_name])
        entries.append(
            FormEntry(
                key=f"dict:{info.cls_name}",
                config_name=info.name,
                cls_name=info.cls_name,
                title=f"{fname} — field",
                schema=jsonforms_schema(dto.json_schema),
                defaults=dto.defaults,
                state_key=js_identifier(f"form_{info.name}"),
                kind="dict",
                step="fields",
                owner_model=owner.get(info.cls_name),
                cls=cfgs[info.cls_name],
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
