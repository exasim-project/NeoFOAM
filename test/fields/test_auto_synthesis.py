# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for :mod:`neofoam.fields.synthesis` and the ``run_build()`` path.

Covers the framework-side synthesis of :class:`InitStep` objects from
:class:`FieldDecl` declarations. The pybFoam dispatch fires only when
the synthesised factory is *called*; these tests stop short of that
(no mesh available) and assert on the synthesised step's name /
depends_on / write / category.
"""

from __future__ import annotations

import pytest

from neofoam.fields.bc import FixedValueBC, NoSlipBC
from neofoam.fields.decl import FieldDecl
from neofoam.fields.synthesis import synthesize_init_step
from neofoam.fields.value_types import Scalar, Vector
from neofoam.framework.model.runtime import ModelRuntime
from neofoam.framework.model.spec import Model


# -- synthesize_init_step --------------------------------------------


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


# -- ModelRuntime.run_build() ----------------------------------------


def test_run_build_auto_synthesizes_for_every_decl() -> None:
    """A bare spec with only field declarations gets a full step list."""
    spec = Model("synth")
    spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC],
        write=True,
    )
    spec.field(
        "p",
        dimensions=[0, 2, -2, 0, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[NoSlipBC],
    )

    @spec.build
    def _build() -> list[object]:  # noqa: ANN401
        return []

    rt = ModelRuntime(spec=spec, name="synth", config=None)
    steps = rt.run_build()
    names = [s.name for s in steps]
    assert names == ["fields.U", "fields.p"]
    assert steps[0].write is True
    assert steps[1].write is False


def test_run_build_prepends_field_steps_before_user_steps() -> None:
    """Auto-synthesised steps come first; @build steps follow.

    The topological sort is the source of truth for execution order at
    run time — order in the returned list is only cosmetic. The
    contract this test pins is: a downstream consumer that walks the
    list in order sees fields first, then user-written infrastructure.
    """
    from neofoam.framework.initialization import lazy

    spec = Model("synth_mixed")
    spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC],
    )

    @spec.build
    def _build() -> list[object]:
        return [lazy("control_object", lambda _ctx: None)]

    rt = ModelRuntime(spec=spec, name="synth_mixed", config=None)
    steps = rt.run_build()
    assert [s.name for s in steps] == ["fields.U", "control_object"]


def test_run_build_handles_specs_without_field_decls() -> None:
    """A spec with no Model.field declarations behaves like before."""
    from neofoam.framework.initialization import lazy

    spec = Model("no_fields")

    @spec.build
    def _build() -> list[object]:
        return [lazy("just_a_thing", lambda _ctx: None)]

    rt = ModelRuntime(spec=spec, name="no_fields", config=None)
    steps = rt.run_build()
    assert [s.name for s in steps] == ["just_a_thing"]


def test_run_build_handles_specs_without_build_func() -> None:
    """No @build but declared fields: auto-synthesis still fires."""
    spec = Model("only_fields")
    spec.field(
        "T",
        dimensions=[0, 0, 0, 1, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[NoSlipBC],
    )
    rt = ModelRuntime(spec=spec, name="only_fields", config=None)
    steps = rt.run_build()
    assert [s.name for s in steps] == ["fields.T"]
