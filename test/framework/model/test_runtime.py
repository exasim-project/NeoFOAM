# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for :class:`ModelRuntime` — the per-instance mutable state.

A ``ModelRuntime`` holds one model instance's loaded config and delegates
every stage (resolve, build, operations) back to its immutable
``ModelSpec`` so state never leaks between solver runs. These tests build
the runtime directly against a real ``ModelSpec`` — with stage functions
registered through ``@spec.resolve``/``@spec.build``/``@spec.operation`` —
and exercise each delegating method in isolation.
"""

from types import SimpleNamespace
from typing import Any

from neofoam.framework.model import ModelRuntime, ModelSpec
from neofoam.io import BaseConfig


def test_model_runtime_stores_name() -> None:
    spec = ModelSpec("HeatSource")
    rt = ModelRuntime(spec=spec, name="HeatSource_zoneA", config={"power": 100})
    assert rt.name == "HeatSource_zoneA"


def test_model_runtime_stores_config() -> None:
    spec = ModelSpec("HeatSource")
    rt = ModelRuntime(spec=spec, name="HeatSource_zoneA", config={"power": 100})
    assert rt.config == {"power": 100}


def test_model_runtime_stores_spec() -> None:
    spec = ModelSpec("HeatSource")
    rt = ModelRuntime(spec=spec, name="HeatSource_zoneA", config={"power": 100})
    assert rt.spec is spec


def test_run_resolve_with_no_resolve_func_leaves_config_unchanged() -> None:
    spec = ModelSpec("M")
    rt = ModelRuntime(spec=spec, name="M_id", config={"v": 1})
    rt.run_resolve(ctx=SimpleNamespace())  # type: ignore[arg-type]
    assert rt.config == {"v": 1}


def test_run_resolve_stores_returned_config() -> None:
    spec = ModelSpec("M")
    new_cfg = {"v": 99}

    @spec.resolve
    def resolve(config: Any, ctx: Any) -> Any:
        return new_cfg

    rt = ModelRuntime(spec=spec, name="M_id", config={"v": 1})
    rt.run_resolve(ctx=SimpleNamespace())  # type: ignore[arg-type]
    assert rt.config is new_cfg


def test_run_resolve_ignores_none_return_and_keeps_original() -> None:
    spec = ModelSpec("M")

    @spec.resolve
    def resolve(config: Any, ctx: Any) -> Any:
        return None

    original = {"v": 1}
    rt = ModelRuntime(spec=spec, name="M_id", config=original)
    rt.run_resolve(ctx=SimpleNamespace())  # type: ignore[arg-type]
    assert rt.config is original


def test_run_build_returns_empty_list_when_no_build_func() -> None:
    spec = ModelSpec("M")
    rt = ModelRuntime(spec=spec, name="M_id", config={})
    assert rt.run_build() == []


def test_run_build_passes_config_to_build_func() -> None:
    spec = ModelSpec("M")
    captured: dict[str, Any] = {}

    @spec.build
    def build(config: Any) -> list[Any]:
        captured["config"] = config
        return []

    rt = ModelRuntime(spec=spec, name="M_id", config={"x": 7})
    rt.run_build()
    assert captured["config"] == {"x": 7}


def test_run_build_returns_build_func_result() -> None:
    spec = ModelSpec("M")

    @spec.build
    def build(config: Any) -> list[Any]:
        return ["step_a", "step_b"]

    rt = ModelRuntime(spec=spec, name="M_id", config={"x": 7})
    assert rt.run_build() == ["step_a", "step_b"]


def test_operations_builds_one_operation_per_registered_operation() -> None:
    """``ModelRuntime.operations`` delegates to ``spec._build_operations_for``."""
    spec = ModelSpec("M")

    @spec.operation()
    def step(self: Any, ctx: Any) -> None:
        pass

    rt = ModelRuntime(spec=spec, name="M_id", config={})
    assert len(rt.operations) == 1


def test_runtime_configs_returns_single_base_config() -> None:
    """When config is a BaseConfig, configs property returns a one-element list."""

    class MyConfig(BaseConfig):
        v: int = 1

    spec = ModelSpec("M")
    cfg = MyConfig()
    rt = ModelRuntime(spec=spec, name="rt", config=cfg)

    assert rt.configs == [cfg]


def test_runtime_configs_returns_all_from_namespace() -> None:
    """When config is a SimpleNamespace of BaseConfigs, all are returned."""

    class ConfigA(BaseConfig):
        a: int = 1

    class ConfigB(BaseConfig):
        b: int = 2

    spec = ModelSpec("M")
    ca, cb = ConfigA(), ConfigB()
    rt = ModelRuntime(spec=spec, name="rt", config=SimpleNamespace(a=ca, b=cb))

    assert sorted(rt.configs, key=id) == sorted([ca, cb], key=id)


def test_runtime_configs_empty_for_non_config() -> None:
    """When config is not a BaseConfig or namespace, configs returns []."""
    spec = ModelSpec("M")
    rt = ModelRuntime(spec=spec, name="rt", config={"plain": "dict"})
    assert rt.configs == []
