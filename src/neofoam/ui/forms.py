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
from neofoam.ui.form_schema import alternatives_uischema, humanize, jsonforms_schema

__all__ = [
    "FormEntry",
    "build_forms",
    "build_field_forms",
    "build_mesh_forms",
    "exclusive_model_families",
    "js_identifier",
    "jsonforms_schema",
    "schema_key",
    "uischema_key",
]

_BC_KEYS = ("boundaryField",)

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


def schema_key(entry: FormEntry) -> str:
    """The wizard state var holding ``entry``'s schema, safe as a JS identifier.

    Use it (never a hand-built name) wherever the schema is read or patched, e.g.
    ``state[schema_key(entry)]``; step plugins get it as ``ctx.schema_key``.
    """
    return js_identifier("schema_" + entry.key)


def uischema_key(entry: FormEntry) -> str:
    """The wizard state var holding ``entry``'s UISchema, safe as a JS identifier.

    The companion of :func:`schema_key`, e.g. ``state[uischema_key(entry)]``.
    """
    return js_identifier("uischema_" + entry.key)


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


def js_identifier(name: str) -> str:
    """``name`` as a JS identifier: trame state vars are evaluated as Vue expressions,

    so a field like ``alpha.water`` would otherwise read as ``alpha`` . ``water…``.
    """
    return re.sub(r"\W", "_", name)


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
                    schema=jsonforms_schema(slice_schema(dto.json_schema, _BC_KEYS)),
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
