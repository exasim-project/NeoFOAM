# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the case-free ``configurations`` schema API on SolverSpec.

Locks in, using fake (pybFoam-free) model families:
- ``models(..., required=...)`` binds a family on the spec.
- ``model_specs`` unions every member of every bound family, case-free.
- ``configurations(solver)`` collects the solver's own configs plus every
  family member's configs, deduped and order-preserved.
- The ``Configurations`` view: ``names`` / iteration / ``__getitem__`` /
  ``new`` / ``json_schema`` / ``as_output_model``.
- ``detect_required_models`` / ``detect_optional_models`` run the family
  detection contracts.
"""

from pathlib import Path
from typing import Any

import pytest

from neofoam.fields.bc import FixedValueBC, GenericBC, NoSlipBC
from neofoam.fields.schema import schema_for
from neofoam.fields.value_types import Scalar, Vector
from neofoam.framework.model import Model
from neofoam.framework.solver import Configurations, Solver, configurations
from neofoam.framework.solver.configurations import _is_field_schema
from neofoam.io import BaseConfig
from neofoam.io.decorator import IOStrategy, OF


class SolverCfgA(BaseConfig):
    a: int = 1


class SolverCfgB(BaseConfig):
    b: str = "x"


class CoreMemberCfg(BaseConfig):
    iters: int = 3


class OptionalMemberCfg(BaseConfig):
    beta: float = 0.5


# A core member spec and a fake core family that owns it.
core_member = Model("CoreMember")
core_member.config(CoreMemberCfg)


class FakeCoreFamily:
    @classmethod
    def all_specs(cls) -> list[Any]:
        return [core_member]

    @classmethod
    def detect_and_create(cls) -> Any:
        return core_member


# An optional member spec and a fake optional family that owns it.
optional_member = Model("OptionalMember")
optional_member.config(OptionalMemberCfg)


class FakeOptionalFamily:
    @classmethod
    def all_specs(cls) -> list[Any]:
        return [optional_member]

    @classmethod
    def detect_models(cls, case_dir: Any = None) -> list[Any]:
        return [optional_member]


class FakeEmptyOptionalFamily:
    @classmethod
    def all_specs(cls) -> list[Any]:
        return [optional_member]

    @classmethod
    def detect_models(cls, case_dir: Any = None) -> list[Any]:
        return []


def _solver() -> Any:
    spec = Solver("fake")
    spec.config(SolverCfgA)
    spec.config(SolverCfgB)
    spec.models(FakeCoreFamily, required=True)
    spec.models(FakeOptionalFamily)
    return spec


def test_required_models_bind_once() -> None:
    spec = _solver()
    spec.models(FakeCoreFamily, required=True)  # idempotent
    assert spec.required_model_specs == [FakeCoreFamily]


def test_optional_models_bind_once() -> None:
    spec = _solver()
    spec.models(FakeOptionalFamily)  # idempotent
    assert spec.optional_model_specs == [FakeOptionalFamily]


def test_models_conflicting_required_flag_raises() -> None:
    spec = Solver("conflict")
    spec.models(FakeCoreFamily, required=True)
    spec.models(FakeCoreFamily, required=True)  # same flag → idempotent no-op
    assert spec.required_model_specs == [FakeCoreFamily]
    assert spec.optional_model_specs == []
    with pytest.raises(ValueError):
        spec.models(FakeCoreFamily, required=False)


def test_detect_optional_models_drops_empty_family() -> None:
    spec = Solver("empty_detect")
    spec.models(FakeEmptyOptionalFamily)
    assert spec.optional_model_specs == [FakeEmptyOptionalFamily]
    assert spec.detect_optional_models(Path(".")) == []


def test_model_specs_unions_all_family_members() -> None:
    spec = _solver()
    assert spec.model_specs == [core_member, optional_member]


def test_configurations_collects_solver_and_member_configs() -> None:
    cfg = configurations(_solver())
    assert isinstance(cfg, Configurations)
    assert cfg.names == [
        "SolverCfgA",
        "SolverCfgB",
        "CoreMemberCfg",
        "OptionalMemberCfg",
    ]
    assert len(cfg) == 4
    assert [c.__name__ for c in cfg] == cfg.names


def test_getitem_returns_config_class_or_raises() -> None:
    cfg = configurations(_solver())
    assert cfg["CoreMemberCfg"] is CoreMemberCfg
    with pytest.raises(KeyError):
        cfg["Missing"]


def test_new_constructs_a_config_instance() -> None:
    cfg = configurations(_solver())
    inst = cfg.new("CoreMemberCfg", iters=9)
    assert inst.iters == 9


def test_json_schema_keys_match_config_names() -> None:
    cfg = configurations(_solver())
    schema = cfg.json_schema()
    assert set(schema) == set(cfg.names)


def test_output_model_fields_are_snake_cased_config_names() -> None:
    cfg = configurations(_solver())
    out = cfg.as_output_model()
    assert set(out.model_fields) == {
        "solver_cfg_a",
        "solver_cfg_b",
        "core_member_cfg",
        "optional_member_cfg",
    }


def test_detect_required_models_runs_the_family_contract() -> None:
    assert _solver().detect_required_models() == [core_member]


def test_detect_optional_models_runs_the_family_contract() -> None:
    assert _solver().detect_optional_models(Path(".")) == [optional_member]


def test_labeled_sets_display_label() -> None:
    assert Model("plain").label == "plain"
    assert Model("buoyancy").labeled("Buoyancy").label == "Buoyancy"


def test_model_catalog_splits_required_and_optional() -> None:
    from neofoam.framework.solver.configurations import ModelEntry, model_catalog

    class CoreCfg(BaseConfig):
        c: int = 1

    class OptCfg(BaseConfig):
        o: int = 2

    core = Model("CoreModel")
    core.config(CoreCfg)
    opt = Model("OptModel").labeled("Opt")
    opt.config(OptCfg)

    class CoreFam:
        @classmethod
        def all_specs(cls) -> list[Any]:
            return [core]

        @classmethod
        def detect_and_create(cls) -> Any:
            return core

    class OptFam:
        @classmethod
        def all_specs(cls) -> list[Any]:
            return [opt]

        @classmethod
        def detect_models(cls, case_dir: Any = None) -> list[Any]:
            return [opt]

    spec = Solver("catalog_solver")
    spec.models(CoreFam, required=True)
    spec.models(OptFam)

    cat = {e.name: e for e in model_catalog(spec)}
    assert isinstance(cat["CoreModel"], ModelEntry)
    assert cat["CoreModel"].required is True
    assert [c.__name__ for c in cat["CoreModel"].dicts] == ["CoreCfg"]
    assert cat["OptModel"].required is False
    assert cat["OptModel"].label == "Opt"
    assert [c.__name__ for c in cat["OptModel"].dicts] == ["OptCfg"]


def test_incompressible_fluid_models_required_vs_optional() -> None:
    from neofoam.framework.solver.configurations import model_catalog
    from neofoam.solver.incompressibleFluid.incompressibleFluid import (
        incompressibleFluid,
    )

    cat = {e.name: e for e in model_catalog(incompressibleFluid)}
    # Core families are required; buoyancy + time-step contributions are optional.
    assert cat["Pimple"].required and cat["Newtonian"].required
    assert cat["laminar"].required
    assert cat["boussinesq"].required is False
    assert cat["courant"].required is False
    assert cat["maxDeltaT"].required is False


def test_incompressible_fluid_boussinesq_label_in_catalog() -> None:
    from neofoam.framework.solver.configurations import model_catalog
    from neofoam.solver.incompressibleFluid.incompressibleFluid import (
        incompressibleFluid,
    )

    cat = {e.name: e for e in model_catalog(incompressibleFluid)}
    bouss = cat["boussinesq"]
    assert bouss.required is False
    assert bouss.label == "Buoyancy (Boussinesq)"
    assert {c.__name__ for c in bouss.dicts} == {
        "BoussinesqConfig",
        "GravityConfig",
        "boussinesq_fvSchemes",
        "boussinesq_fvSolution",
    }
    assert {c.__name__ for c in bouss.fields} == {
        "p_rghFieldConfig",
        "TFieldConfig",
        "alphatFieldConfig",
    }


# ===========================================================================
# Field-schema surface of ``configurations(...)``
#
# A synthetic solver (one model spec, two declared fields) exercises the
# field-schema iteration without dragging the full incompressibleFluid stack;
# the real solver wiring is exercised in ``test/fields/test_in_tree_models.py``.
# ===========================================================================


class _SyntheticSolver:
    """Minimal solver duck-type accepted by ``configurations(solver)``.

    Mirrors the attributes ``collect_config_classes`` reads:
    ``_config_classes`` for dictionary configs and ``model_specs`` for the
    model registry. No real solver behaviour is exercised — the point is to
    keep these tests independent of the full SolverSpec.
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
