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
from neofoam.framework.tools import Tool
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


class PreprocessMemberCfg(BaseConfig):
    pipeline: list = []


preprocess_tool = Tool("PreprocessTool")
preprocess_tool.config(PreprocessMemberCfg)


def _solver() -> Any:
    spec = Solver("fake")
    spec.config(SolverCfgA)
    spec.config(SolverCfgB)
    spec.core_models(FakeCoreFamily)
    spec.optional_models(FakeOptionalFamily)
    return spec


def test_configurations_includes_preprocess_tool() -> None:
    spec = _solver()
    spec.tools(preprocess_tool)
    cfg = configurations(spec)
    assert "PreprocessMemberCfg" in cfg.names


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
