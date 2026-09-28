# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for :class:`neofoam.fields.decl.FieldDecl`.

Asserts the immutable record holds what was declared. Runtime
construction of the matching :class:`InitStep` is the framework's job
(see :mod:`neofoam.fields.synthesis`); those tests live in
``test_auto_synthesis.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from neofoam.fields.bc import FixedValueBC, NoSlipBC
from neofoam.fields.decl import FieldDecl
from neofoam.fields.value_types import Vector


def _U_decl() -> FieldDecl:
    return FieldDecl(
        name="U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=(NoSlipBC, FixedValueBC),
        write=True,
        depends_on=("mesh",),
    )


def test_field_decl_records_inputs() -> None:
    decl = _U_decl()
    assert decl.name == "U"
    assert decl.dimensions == [0, 1, -1, 0, 0, 0, 0]
    assert decl.value_type is Vector
    assert decl.allowed_bcs == (NoSlipBC, FixedValueBC)
    assert decl.write is True
    assert decl.depends_on == ("mesh",)
    assert decl.initial_value is None


def test_field_decl_is_immutable() -> None:
    decl = _U_decl()
    with pytest.raises(dataclasses.FrozenInstanceError):
        decl.name = "V"  # type: ignore[misc]
