# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for ``ModelSpec.field(...)`` registration.

Asserts the registered :class:`FieldDecl` is exposed via
``spec.field_decls``, the declaration round-trip is intact, and a
duplicate name on the same spec is an error.
"""

from __future__ import annotations

import pytest

from neofoam.fields.bc import FixedValueBC, NoSlipBC
from neofoam.fields.decl import FieldDecl
from neofoam.fields.value_types import Vector
from neofoam.framework.model.spec import Model


def test_field_returns_decl_handle() -> None:
    spec = Model("Test")
    decl = spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC, FixedValueBC],
        write=True,
    )
    assert isinstance(decl, FieldDecl)
    assert decl.name == "U"
    assert decl.value_type is Vector
    assert decl.allowed_bcs == (NoSlipBC, FixedValueBC)
    assert decl.write is True
    assert decl.depends_on == ("mesh",)


def test_field_decls_lists_declarations_in_order() -> None:
    spec = Model("Test")
    U = spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC],
    )
    p = spec.field(
        "p",
        dimensions=[0, 2, -2, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[FixedValueBC],
    )
    assert spec.field_decls == (U, p)


def test_duplicate_field_name_is_error() -> None:
    spec = Model("Test")
    spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC],
    )
    with pytest.raises(ValueError, match="field 'U' already declared"):
        spec.field(
            "U",
            dimensions=[0, 1, -1, 0, 0, 0, 0],
            value_type=Vector,
            allowed_bcs=[NoSlipBC],
        )
