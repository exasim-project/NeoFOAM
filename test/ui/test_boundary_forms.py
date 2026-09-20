# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for seeding and pinning boundary conditions from patches."""

from __future__ import annotations

import pytest

from neofoam.ui.boundary_forms import seed_boundary_field
from neofoam.ui.forms import build_forms


def test_hand_added_patch_starts_as_a_bc_object(solver):
    # JSONForms seeds a hand-added key from its schema's `type`; the BC union declares
    # none, so the new patch became the string "" and rendered as a broken text box.
    entries = {e.key: e for e in build_forms(solver)}
    boundary_field = entries["field_bc:UFieldConfig"].schema["properties"]["boundaryField"]
    assert boundary_field["additionalProperties"]["type"] == "object"


@pytest.mark.parametrize(
    ("entry_key", "seed"),
    [
        ("field_bc:UFieldConfig", {"type": "noSlip"}),
        ("field_bc:pFieldConfig", {"type": "zeroGradient"}),
        ("field_bc:alphatFieldConfig", {"type": "fixedValue", "value": "uniform 0"}),
    ],
)
def test_hand_added_patch_starts_as_the_wall_bc_a_scan_would_seed(entry_key, seed, solver):
    # JSONForms seeds a hand-added key from the schema `default`. Without one the patch
    # was `{}`: the row showed the GenericBC fallback, and the save turned it into the
    # union's first all-default arm without saying so.
    entries = {e.key: e for e in build_forms(solver)}
    boundary_field = entries[entry_key].schema["properties"]["boundaryField"]
    assert boundary_field["additionalProperties"]["default"] == seed
    scanned = seed_boundary_field(entries[entry_key], [{"name": "walls", "role": "wall"}], {})
    assert scanned["boundaryField"]["walls"] == seed


def test_allowed_bc_types_lists_titled_arms(solver):
    from neofoam.ui.boundary_forms import allowed_bc_types  # noqa: PLC0415

    entries = {e.key: e for e in build_forms(solver)}
    u = allowed_bc_types(entries["field_bc:UFieldConfig"])
    p = allowed_bc_types(entries["field_bc:pFieldConfig"])
    assert {"noSlip", "fixedValue", "zeroGradient", "empty"} <= set(u)
    assert "noSlip" in u and "noSlip" not in p  # vector-only arm


def test_seed_boundary_field_roles_values_and_preservation(solver):
    from neofoam.ui.boundary_forms import seed_boundary_field  # noqa: PLC0415

    entries = {e.key: e for e in build_forms(solver)}
    patches = [
        {"name": "inlet", "role": "inlet"},
        {"name": "outlet", "role": "outlet"},
        {"name": "walls", "role": "wall"},
        {"name": "frontBack", "role": "empty"},
    ]
    u = seed_boundary_field(entries["field_bc:UFieldConfig"], patches, {})["boundaryField"]
    # Role → BC type, clamped to the field's allowed arms. Role `empty` seeds a
    # symmetry BC — the snappy mesh realises empty patches as symmetry patches.
    assert u["frontBack"] == {"type": "symmetry"}
    assert u["walls"] == {"type": "noSlip"}
    assert u["outlet"] == {"type": "zeroGradient"}
    # Value-carrying types are seeded with a valid zero payload (matching the arm).
    assert u["inlet"] == {"type": "fixedValue", "value": "uniform (0 0 0)"}
    # A scalar field gets the scalar zero literal.
    p = seed_boundary_field(entries["field_bc:pFieldConfig"], patches, {})["boundaryField"]
    assert p["inlet"] == {"type": "fixedValue", "value": "uniform 0"}
    # Existing entries are preserved (not overwritten).
    keep = {"boundaryField": {"inlet": {"type": "fixedValue", "value": "uniform (1 0 0)"}}}
    out = seed_boundary_field(entries["field_bc:UFieldConfig"], patches, keep)
    assert out["boundaryField"]["inlet"]["value"] == "uniform (1 0 0)"


def test_patch_bc_schema_pins_named_patch_sections(solver):
    from neofoam.ui.boundary_forms import patch_bc_schema  # noqa: PLC0415

    entries = {e.key: e for e in build_forms(solver)}
    entry = entries["field_bc:UFieldConfig"]
    schema = patch_bc_schema(entry, ["inlet", "walls"])
    bf = schema["properties"]["boundaryField"]
    assert bf["type"] == "object"
    assert set(bf["properties"]) == {"inlet", "walls"}
    assert bf["properties"]["inlet"]["title"] == "inlet"
    assert "oneOf" in bf["properties"]["inlet"]  # each patch keeps the BC-type union
    assert "$defs" not in schema  # arms are inlined — nothing left to resolve
    # No names → schema returned unchanged.
    assert patch_bc_schema(entry, []) is entry.schema


def test_patch_bc_schema_keeps_the_patch_adder(solver):
    # A geometry scan only knows the STL patches; one the mesh adds otherwise (a
    # blockMesh face, a baffle) is still added by hand afterwards.
    from neofoam.ui.boundary_forms import patch_bc_schema  # noqa: PLC0415

    entries = {e.key: e for e in build_forms(solver)}
    entry = entries["field_bc:UFieldConfig"]
    before = entry.schema["properties"]["boundaryField"]
    after = patch_bc_schema(entry, ["inlet", "wall.left"])["properties"]["boundaryField"]
    assert after["additionalProperties"] == before["additionalProperties"]
    assert after["i18n"] == "nf.patch"
    assert after["nfPatches"] is True
    assert set(after["properties"]) == {"inlet", "wall.left"}
