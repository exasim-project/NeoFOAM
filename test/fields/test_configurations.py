# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the ``configurations(solver)`` extension to surface field schemas.

A synthetic solver (one core model spec, two declared fields) exercises
the iteration without dragging the full incompressibleFluid stack — the
real solver wiring is exercised in ``test_loader.py``.
"""

from __future__ import annotations

import pytest

from neofoam.fields.bc import FixedValueBC, GenericBC, NoSlipBC
from neofoam.fields.value_types import Scalar, Vector
from neofoam.framework.model.spec import Model
from neofoam.framework.solver.configurations import _is_field_schema, configurations
from neofoam.io.base import BaseConfig
from neofoam.io.decorator import IOStrategy, OF


class _SyntheticSolver:
    """Minimal solver duck-type accepted by ``configurations(solver)``.

    Mirrors the attributes ``collect_config_classes`` reads:
    ``_config_classes`` for dictionary configs and ``model_specs`` for
    the model registry. No real solver behaviour is exercised — the
    point is to keep this test independent of the full SolverSpec.
    """

    def __init__(self, name: str, model_specs: list[object]) -> None:
        self.name = name
        self._config_classes: list[type] = []
        self.model_specs = model_specs


@IOStrategy(OF("constant/dummyDict"))
class _DummyDict(BaseConfig):
    """A plain ``constant/`` config to verify it co-exists with field schemas."""

    setting: int = 0


def _solver_with_two_fields() -> _SyntheticSolver:
    spec = Model("synth")
    spec.config(_DummyDict)
    spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC, FixedValueBC],
        write=True,
    )
    spec.field(
        "T",
        dimensions=[0, 0, 0, 1, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[FixedValueBC, GenericBC],
    )
    return _SyntheticSolver("synth", [spec])


def test_configurations_includes_field_schemas() -> None:
    cfgs = configurations(_solver_with_two_fields())
    names = cfgs.names
    assert "_DummyDict" in names
    assert "UFieldConfig" in names
    assert "TFieldConfig" in names


def test_fields_filter_returns_only_field_schemas() -> None:
    cfgs = configurations(_solver_with_two_fields())
    field_names = [cls.__name__ for cls in cfgs.fields]
    assert field_names == ["UFieldConfig", "TFieldConfig"]
    for cls in cfgs.fields:
        assert cls.io_config is not None
        assert cls.io_config.file.startswith("0/")


def test_dicts_filter_excludes_field_schemas() -> None:
    cfgs = configurations(_solver_with_two_fields())
    dict_names = [cls.__name__ for cls in cfgs.dicts]
    assert "_DummyDict" in dict_names
    assert all(not cls.io_config.file.startswith("0/") for cls in cfgs.dicts)


def test_is_field_schema_dispatch() -> None:
    assert _is_field_schema(_DummyDict) is False

    spec = Model("synth")
    U_decl = spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC],
    )
    from neofoam.fields.schema import schema_for

    assert _is_field_schema(schema_for(U_decl)) is True


def test_repeated_call_returns_referentially_stable_schemas() -> None:
    """Schema synthesis is cached per FieldDecl — two calls = same class."""
    solver = _solver_with_two_fields()
    a = configurations(solver).fields
    b = configurations(solver).fields
    assert a == b  # same classes, same order
    for ca, cb in zip(a, b, strict=True):
        assert ca is cb


def test_configurations_lookup_by_name() -> None:
    cfgs = configurations(_solver_with_two_fields())
    Cls = cfgs["UFieldConfig"]
    assert Cls.io_config is not None
    assert Cls.io_config.file == "0/U"

    with pytest.raises(KeyError):
        cfgs["nonexistent"]
