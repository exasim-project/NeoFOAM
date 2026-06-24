# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the case-free ``configurations`` schema API on SolverSpec.

Locks in, using fake (pybFoam-free) model families:
- ``core_models`` / ``optional_models`` bind a family on the spec.
- ``model_specs`` unions every member of every bound family, case-free.
- ``configurations(solver)`` collects the solver's own configs plus every
  family member's configs, deduped and order-preserved.
- The ``Configurations`` view: ``names`` / iteration / ``__getitem__`` /
  ``new`` / ``json_schema`` / ``as_output_model``.
- ``detect_core_models`` / ``detect_optional_models`` run the family
  detection contracts.
"""

from pathlib import Path
from typing import Any

import pytest

from neofoam.framework.model import Model
from neofoam.framework.solver import Configurations, Solver, configurations
from neofoam.io import BaseConfig


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


def _solver() -> Any:
    spec = Solver("fake")
    spec.config(SolverCfgA)
    spec.config(SolverCfgB)
    spec.core_models(FakeCoreFamily)
    spec.optional_models(FakeOptionalFamily)
    return spec


def test_core_and_optional_models_bind_once() -> None:
    spec = _solver()
    spec.core_models(FakeCoreFamily)  # idempotent
    spec.optional_models(FakeOptionalFamily)
    assert spec._core_model_specs == [FakeCoreFamily]
    assert spec._optional_model_specs == [FakeOptionalFamily]


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


def test_getitem_and_new() -> None:
    cfg = configurations(_solver())
    assert cfg["CoreMemberCfg"] is CoreMemberCfg
    inst = cfg.new("CoreMemberCfg", iters=9)
    assert inst.iters == 9
    with pytest.raises(KeyError):
        cfg["Missing"]


def test_json_schema_and_output_model() -> None:
    cfg = configurations(_solver())
    schema = cfg.json_schema()
    assert set(schema) == set(cfg.names)

    out = cfg.as_output_model()
    assert set(out.model_fields) == {
        "solver_cfg_a",
        "solver_cfg_b",
        "core_member_cfg",
        "optional_member_cfg",
    }


def test_detect_core_and_optional_models() -> None:
    spec = _solver()
    assert spec.detect_core_models() == [core_member]
    assert spec.detect_optional_models(Path(".")) == [optional_member]


# -- toggle models ----------------------------------------------------


def test_as_toggle_sets_flag_and_label() -> None:
    m = Model("buoyancy").as_toggle("Buoyancy")
    assert m.toggle is True
    assert m.toggle_label == "Buoyancy"
    # Defaults to the model name when no label is given.
    assert Model("plain").as_toggle().toggle_label == "plain"
    # Unflagged models default to off.
    assert Model("off").toggle is False


def test_toggle_models_lists_flagged_optional_with_owned_configs() -> None:
    from neofoam.framework.solver.configurations import ToggleModel, toggle_models

    class ToggleCfg(BaseConfig):
        on: bool = True

    member = Model("Buoyancy").as_toggle("Buoyancy")
    member.config(ToggleCfg)

    class ToggleFamily:
        @classmethod
        def all_specs(cls) -> list[Any]:
            return [member]

        @classmethod
        def detect_models(cls, case_dir: Any = None) -> list[Any]:
            return [member]

    spec = Solver("toggle_solver")
    spec.optional_models(ToggleFamily)

    tms = toggle_models(spec)
    assert len(tms) == 1
    t = tms[0]
    assert isinstance(t, ToggleModel)
    assert (t.name, t.label) == ("Buoyancy", "Buoyancy")
    assert [c.__name__ for c in t.dicts] == ["ToggleCfg"]
    assert t.fields == []


def test_toggle_models_skips_unflagged_optional() -> None:
    from neofoam.framework.solver.configurations import toggle_models

    # _solver()'s optional member is not flagged → no toggles.
    assert toggle_models(_solver()) == []


def test_model_catalog_splits_required_and_optional() -> None:
    from neofoam.framework.solver.configurations import ModelEntry, model_catalog

    class CoreCfg(BaseConfig):
        c: int = 1

    class OptCfg(BaseConfig):
        o: int = 2

    core = Model("CoreModel")
    core.config(CoreCfg)
    opt = Model("OptModel").as_toggle("Opt")
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
    spec.core_models(CoreFam)
    spec.optional_models(OptFam)

    cat = {e.name: e for e in model_catalog(spec)}
    assert isinstance(cat["CoreModel"], ModelEntry)
    assert cat["CoreModel"].required is True
    assert [c.__name__ for c in cat["CoreModel"].dicts] == ["CoreCfg"]
    assert cat["OptModel"].required is False
    assert cat["OptModel"].label == "Opt"
    assert [c.__name__ for c in cat["OptModel"].dicts] == ["OptCfg"]


def test_incompressible_fluid_models_required_vs_optional() -> None:
    pytest.importorskip("pybFoam")
    from neofoam.framework.solver.configurations import model_catalog
    from neofoam.solver.incompressibleFluid.incompressibleFluid import (
        incompressibleFluid,
    )

    cat = {e.name: e for e in model_catalog(incompressibleFluid)}
    # Core families are required; buoyancy + adaptive stepping are optional.
    assert cat["Pimple"].required and cat["Newtonian"].required
    assert cat["laminar"].required
    assert cat["boussinesq"].required is False
    assert cat["adaptiveTimeStep"].required is False


def test_incompressible_fluid_buoyancy_is_a_toggle() -> None:
    pytest.importorskip("pybFoam")
    from neofoam.framework.solver.configurations import toggle_models
    from neofoam.solver.incompressibleFluid.incompressibleFluid import (
        incompressibleFluid,
    )

    tms = {t.name: t for t in toggle_models(incompressibleFluid)}
    assert "boussinesq" in tms
    bouss = tms["boussinesq"]
    assert bouss.label == "Buoyancy (Boussinesq)"
    assert {c.__name__ for c in bouss.dicts} == {
        "BoussinesqConfig",
        "boussinesq_fvSchemes",
        "boussinesq_fvSolution",
    }
    assert {c.__name__ for c in bouss.fields} == {
        "p_rghFieldConfig",
        "TFieldConfig",
        "alphatFieldConfig",
    }


def test_incompressible_fluid_adaptive_time_step_is_a_toggle() -> None:
    pytest.importorskip("pybFoam")
    from neofoam.framework.solver.configurations import toggle_models
    from neofoam.solver.incompressibleFluid.incompressibleFluid import (
        incompressibleFluid,
    )

    tms = {t.name: t for t in toggle_models(incompressibleFluid)}
    assert "adaptiveTimeStep" in tms
    cfl = tms["adaptiveTimeStep"]
    assert cfl.label == "Adaptive time step (Courant)"
    # owns only the controlDict slice; no 0/ fields
    assert {c.__name__ for c in cfl.dicts} == {"CourantControlConfig"}
    assert cfl.fields == []
