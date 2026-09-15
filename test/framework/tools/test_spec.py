# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""ToolSpec API: typed/untyped build seam, instantiate."""

from typing import Any, Literal

import pytest
from pydantic import BaseModel, ValidationError

from neofoam.framework.initialization import InitStep, lazy
from neofoam.framework.tools import Tool, ToolRuntime


class SomeStep(BaseModel):
    tool: Literal["x"]
    k: int = 7


def test_step_config_type_typed() -> None:
    t = Tool("x")

    @t.build
    def _b(cfg: SomeStep) -> list[InitStep]:
        return []

    assert t.step_config_type is SomeStep


def test_step_config_type_untyped_is_none() -> None:
    untyped = Tool("y")

    @untyped.build
    def _b(cfg: Any) -> list[InitStep]:
        return []

    assert untyped.step_config_type is None
    # A tool with no build func registered also has no step-config type.
    assert Tool("z").step_config_type is None


def test_instantiate_typed_validates_and_names() -> None:
    t = Tool("x")

    @t.build
    def _b(cfg: SomeStep) -> list[InitStep]:
        return []

    rt = t.instantiate({"tool": "x"})
    assert isinstance(rt, ToolRuntime)
    assert rt.name == "preprocess.x"
    assert isinstance(rt.config, SomeStep)
    assert rt.config.k == 7  # default applied


def test_instantiate_untyped_passes_raw_mapping() -> None:
    untyped = Tool("y")

    @untyped.build
    def _b(cfg: Any) -> list[InitStep]:
        return []

    entry = {"tool": "y", "k": 1}
    rt = untyped.instantiate(entry)
    assert rt.config == {"tool": "y", "k": 1}


def test_instantiate_typed_invalid_raises() -> None:
    t = Tool("x")

    @t.build
    def _b(cfg: SomeStep) -> list[InitStep]:
        return []

    with pytest.raises(ValidationError):
        t.instantiate({"tool": "x", "k": "notanint"})


def test_instantiate_carries_depends_on() -> None:
    t = Tool("x")

    @t.build
    def _b(cfg: SomeStep) -> list[InitStep]:
        return []

    rt = t.instantiate({"tool": "x", "depends_on": ["block"], "k": 1})
    assert rt.depends_on == ["block"]
    # the depends_on envelope key is ignored by the typed step config
    assert rt.config.k == 1


def test_instantiate_defaults_depends_on_empty() -> None:
    t = Tool("x")

    @t.build
    def _b(cfg: SomeStep) -> list[InitStep]:
        return []

    assert t.instantiate({"tool": "x"}).depends_on == []


def test_runtime_run_build_calls_build() -> None:
    t = Tool("x")
    step = lazy("preprocess.x", lambda ctx: "result")

    @t.build
    def _b(cfg: SomeStep) -> list[InitStep]:
        return [step]

    rt = t.instantiate({"tool": "x"})
    assert rt.run_build() == [step]
