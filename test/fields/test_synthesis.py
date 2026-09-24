# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for :mod:`neofoam.fields.synthesis`.

Covers the framework-side synthesis of an :class:`InitStep` from a
:class:`FieldDecl` declaration. The pybFoam dispatch fires only when the
synthesised factory is *called*; these tests stop short of that (no mesh
available) and assert on the synthesised step's name / depends_on /
write. The ``ModelRuntime.run_build`` integration that drives this
synthesizer lives in ``test/framework/model/test_spec.py``.
"""

from __future__ import annotations

import pytest

from neofoam.fields.bc import FixedValueBC, NoSlipBC
from neofoam.fields.decl import FieldDecl
from neofoam.fields.synthesis import synthesize_init_step
from neofoam.fields.value_types import Vector


def _U_decl(**overrides: object) -> FieldDecl:
    return FieldDecl(
        name=overrides.get("name", "U"),  # type: ignore[arg-type]
        dimensions=overrides.get("dimensions", [0, 1, -1, 0, 0, 0, 0]),  # type: ignore[arg-type]
        value_type=overrides.get("value_type", Vector),  # type: ignore[arg-type]
        allowed_bcs=(NoSlipBC, FixedValueBC),
        write=overrides.get("write", True),  # type: ignore[arg-type]
        depends_on=overrides.get("depends_on", ("mesh",)),  # type: ignore[arg-type]
    )


def test_synthesize_uses_fields_prefix() -> None:
    step = synthesize_init_step(_U_decl())
    assert step.name == "fields.U"


def test_synthesize_propagates_depends_on() -> None:
    step = synthesize_init_step(_U_decl(depends_on=("mesh", "fields.p")))
    assert step.depends_on == ["mesh", "fields.p"]


def test_synthesize_propagates_write_flag() -> None:
    assert synthesize_init_step(_U_decl(write=True)).write is True
    assert synthesize_init_step(_U_decl(write=False)).write is False


def test_synthesize_rejects_unknown_value_type_lazily() -> None:
    """A bogus value_type only blows up when the factory is invoked.

    Lazy resolution lets ``import neofoam.fields.synthesis`` stay light
    (no pybFoam at import time). The check fires inside the factory
    closure — which we trigger here with a dummy mesh.
    """

    class _Bogus:
        pass

    decl = _U_decl(value_type=_Bogus)
    step = synthesize_init_step(decl)
    # Constructing the step is fine — only invocation hits the dispatch.
    with pytest.raises(TypeError, match="no read_field dispatch"):
        step.initializer({"mesh": object()})
