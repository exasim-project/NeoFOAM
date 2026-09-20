# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Seed and pin a field's boundary conditions from the scanned patches (headless, pure)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neofoam.ui.form_schema import _NF_PATCHES

if TYPE_CHECKING:
    from neofoam.ui.forms import FormEntry

__all__ = ["allowed_bc_types", "patch_bc_schema", "seed_boundary_field"]

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
    return _arm_titles(ap)


def _arm_titles(union: dict[str, Any]) -> list[str]:
    """The titles of a union's titled arms, in order."""
    arms = union.get("oneOf") or union.get("anyOf") or []
    return [
        arm["title"] for arm in arms if isinstance(arm, dict) and isinstance(arm.get("title"), str)
    ]


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


def _role_bc_seed(role: str, allowed: list[str]) -> dict[str, Any]:
    """The BC payload a patch of ``role`` starts with, given the field's ``allowed`` types."""
    is_vector = any(marker in allowed for marker in _VECTOR_MARKERS)
    return _bc_seed(_bc_type_for_role(role, allowed), is_vector)


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
    data = dict(current or {})
    bf = dict(data.get("boundaryField") or {})
    for patch in patches:
        name = patch.get("name")
        if name and name not in bf:
            bf[name] = _role_bc_seed(patch.get("role", ""), allowed)
    data["boundaryField"] = bf
    return data


def patch_bc_schema(bc_entry: FormEntry, names: list[str]) -> dict[str, Any]:
    """Rebuild a ``field_bc`` schema with one concrete ``boundaryField`` property per patch.

    The default schema types ``boundaryField`` as an open ``additionalProperties`` map,
    which JSONForms renders as a bare "property name" editor with a generic type picker.
    Once the patch names are known (from the geometry scan) we pin them as titled object
    properties as well, so each scanned patch is a fixed row with a clean BC-type dropdown.
    The map stays open: a patch the scan cannot see is still added by hand. Returns
    ``bc_entry.schema`` unchanged when there are no names or no BC arms.
    """
    base = bc_entry.schema
    bf = base.get("properties", {}).get("boundaryField", {})
    value_schema = bf.get("additionalProperties")
    if not names or not isinstance(value_schema, dict):
        return base
    props = {name: {**value_schema, "title": name} for name in names}
    new_bf = {
        **bf,
        "type": "object",
        "title": bf.get("title", "Boundary field"),
        "properties": props,
    }
    return {
        **base,
        "properties": {**base.get("properties", {}), "boundaryField": new_bf},
    }


def _patch_adder(schema: dict[str, Any]) -> dict[str, Any]:
    """A ``boundaryField`` schema drawn as one row per patch, its "add a key" row adding a patch.

    ``nfPatches`` picks the bundled row renderer, which binds a patch through the map's
    data, so a name holding ``.`` (``wall.left``) works. The row is labelled as a
    patch-name box, and the BC union is typed ``object``: a new key is seeded from the
    ``type``, and with none it would be the string ``""``. Its ``default`` is the BC a
    scan seeds on a wall, so a hand-added patch shows and saves a real type from the start.
    """
    props = schema["properties"]
    boundary_field = {**props["boundaryField"], "i18n": "nf.patch", _NF_PATCHES: True}
    union = boundary_field["additionalProperties"]
    boundary_field["additionalProperties"] = {
        **union,
        "type": "object",
        "default": _role_bc_seed("wall", _arm_titles(union)),
    }
    return {**schema, "properties": {**props, "boundaryField": boundary_field}}
