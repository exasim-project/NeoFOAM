# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Unit tests for the unified operation_wrapper module."""

from types import SimpleNamespace
from typing import Annotated, Any

import pytest

from neofoam.io import BaseConfig
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization.depends import Depends

from framework.components.conftest import MyConfig, OtherConfig, StepConfig


# ===========================================================================
# Cycle 1 — discover_configs_from_signature + find_config_by_type
# ===========================================================================


def test_discover_configs_finds_baseconfig_params() -> None:
    """discover_configs_from_signature returns config params from function signature."""
    from neofoam.framework.operation_wrapper import discover_configs_from_signature

    def my_op(self: Any, cfg: MyConfig, value: float) -> None:
        pass

    result = discover_configs_from_signature(my_op)
    assert len(result) == 1
    assert result[0]["param_name"] == "cfg"
    assert result[0]["config_type"] is MyConfig


def test_discover_configs_skips_self_and_ctx() -> None:
    """discover_configs_from_signature skips 'self' and 'ctx' parameters."""
    from neofoam.framework.operation_wrapper import discover_configs_from_signature

    def my_op(self: Any, ctx: Any) -> None:
        pass

    result = discover_configs_from_signature(my_op)
    assert result == []


def test_discover_configs_finds_multiple_configs() -> None:
    """discover_configs_from_signature finds multiple BaseConfig params."""
    from neofoam.framework.operation_wrapper import discover_configs_from_signature

    def my_op(cfg1: MyConfig, cfg2: StepConfig) -> None:
        pass

    result = discover_configs_from_signature(my_op)
    assert len(result) == 2
    assert result[0]["config_type"] is MyConfig
    assert result[1]["config_type"] is StepConfig


def test_discover_configs_returns_empty_for_no_configs() -> None:
    """discover_configs_from_signature returns [] when no BaseConfig params."""
    from neofoam.framework.operation_wrapper import discover_configs_from_signature

    def my_op(x: float, y: int) -> None:
        pass

    result = discover_configs_from_signature(my_op)
    assert result == []


def test_find_config_by_type_raises_on_missing_type() -> None:
    """find_config_by_type raises ValueError when no match exists."""
    from neofoam.framework.operation_wrapper import find_config_by_type

    cfg = MyConfig()
    with pytest.raises(ValueError, match="OtherConfig"):
        find_config_by_type(cfg, OtherConfig)


def test_find_config_by_type_finds_direct_match() -> None:
    """find_config_by_type returns the config when it directly matches."""
    from neofoam.framework.operation_wrapper import find_config_by_type

    cfg = MyConfig()
    assert find_config_by_type(cfg, MyConfig) is cfg


def test_find_config_by_type_searches_namespace() -> None:
    """find_config_by_type locates a config nested inside a SimpleNamespace."""
    from neofoam.framework.operation_wrapper import find_config_by_type

    ns = SimpleNamespace(step=StepConfig(), main=MyConfig())
    assert isinstance(find_config_by_type(ns, StepConfig), StepConfig)
    assert isinstance(find_config_by_type(ns, MyConfig), MyConfig)


def test_find_config_by_type_raises_when_not_in_namespace() -> None:
    """find_config_by_type raises when config type missing from namespace."""
    from neofoam.framework.operation_wrapper import find_config_by_type

    class Missing(BaseConfig):
        pass

    ns = SimpleNamespace()
    with pytest.raises(ValueError, match="Missing"):
        find_config_by_type(ns, Missing)


# ===========================================================================
# Cycle 2 — DependencyResolver moved to operation_wrapper
# ===========================================================================


def test_dependency_resolver_resolves_depends_marker() -> None:
    """DependencyResolver.resolve_arguments resolves a Depends() marker."""
    from neofoam.framework.operation_wrapper import DependencyResolver

    def provider() -> int:
        return 42

    def my_func(val: Annotated[int, Depends(provider)]) -> int:
        return val

    resolver = DependencyResolver()
    kwargs = resolver.resolve_arguments(my_func)
    assert kwargs["val"] == 42


def test_dependency_resolver_resolves_context_fields() -> None:
    """DependencyResolver.resolve_arguments resolves fields from Context."""
    from neofoam.framework.operation_wrapper import DependencyResolver

    def my_func(x: float) -> float:
        return x

    ctx = Context(fields={"x": 3.14}, models={})
    resolver = DependencyResolver()
    kwargs = resolver.resolve_arguments(my_func, ctx)
    assert kwargs["x"] == 3.14


def test_dependency_resolver_clear_scope() -> None:
    """DependencyResolver.clear_scope clears cached values for a scope."""
    from neofoam.framework.operation_wrapper import DependencyResolver

    resolver = DependencyResolver()
    resolver._cache["time_step"]["key"] = "cached"
    resolver.clear_scope("time_step")
    assert resolver._cache["time_step"] == {}


# ===========================================================================
# Cycle 3 — unified wrap_operation()
# ===========================================================================


def test_wrap_operation_injects_config_when_baseconfig_param() -> None:
    """wrap_operation injects config from runtime.config when func has a BaseConfig param."""
    from neofoam.framework.operation_wrapper import wrap_operation, DependencyResolver

    def my_op(self: Any, cfg: MyConfig) -> None:
        self._got_cfg = cfg

    runtime = SimpleNamespace(config=MyConfig(x=42))
    resolver = DependencyResolver()

    wrapper = wrap_operation(my_op, runtime, resolver)
    ctx = Context(fields={}, models={})
    wrapper(ctx)

    assert runtime._got_cfg.x == 42


def test_wrap_operation_uses_dependency_resolution_when_no_config() -> None:
    """wrap_operation falls through to dependency resolution when no BaseConfig param."""
    from neofoam.framework.operation_wrapper import wrap_operation, DependencyResolver

    def my_op(self: Any, x: float) -> None:
        self._got_x = x

    runtime = SimpleNamespace()
    resolver = DependencyResolver()

    wrapper = wrap_operation(my_op, runtime, resolver)
    ctx = Context(fields={"x": 7.5}, models={})
    wrapper(ctx)

    assert runtime._got_x == 7.5


# ===========================================================================
# Cycle 4 — FieldUpdates handling
# ===========================================================================


def test_wrap_operation_handles_field_updates() -> None:
    """wrap_operation updates ctx.fields when func returns FieldUpdates and returns None."""
    from neofoam.framework.operation_wrapper import wrap_operation, DependencyResolver

    def my_op(self: Any, cfg: MyConfig) -> FieldUpdates:
        return FieldUpdates({"p": 1.0})

    runtime = SimpleNamespace(config=MyConfig())
    resolver = DependencyResolver()

    wrapper = wrap_operation(my_op, runtime, resolver)
    ctx = Context(fields={}, models={})
    result = wrapper(ctx)

    assert result is None
    assert ctx.fields["p"] == 1.0


def test_wrap_operation_handles_field_updates_via_dependency_path() -> None:
    """wrap_operation handles FieldUpdates when using dependency resolution path too."""
    from neofoam.framework.operation_wrapper import wrap_operation, DependencyResolver

    def my_op(self: Any, x: float) -> FieldUpdates:
        return FieldUpdates({"result": x * 2})

    runtime = SimpleNamespace()
    resolver = DependencyResolver()

    wrapper = wrap_operation(my_op, runtime, resolver)
    ctx = Context(fields={"x": 5.0}, models={})
    result = wrapper(ctx)

    assert result is None
    assert ctx.fields["result"] == 10.0


# ===========================================================================
# Cycle 5 — inject_and_call helper
# ===========================================================================


def test_inject_and_call_binds_self_and_injects_config() -> None:
    from neofoam.framework.operation_wrapper import inject_and_call

    class MyCfg(BaseConfig):
        val: int = 42

    captured: dict[str, Any] = {}

    def func(self: Any, cfg: MyCfg) -> list[Any]:
        captured["self"] = self
        captured["cfg"] = cfg
        return ["step"]

    runtime = SimpleNamespace(config=MyCfg(val=99))
    call_meta = {
        "first_param_name": "self",
        "has_ctx": False,
        "config_params": [{"param_name": "cfg", "config_type": MyCfg}],
    }
    result = inject_and_call(func, runtime, call_meta)
    assert captured["self"] is runtime
    assert captured["cfg"].val == 99
    assert result == ["step"]


def test_inject_and_call_passes_ctx() -> None:
    from neofoam.framework.operation_wrapper import inject_and_call

    captured: dict[str, Any] = {}

    def func(self: Any, ctx: Any) -> None:
        captured["ctx"] = ctx

    sentinel = object()
    call_meta = {
        "first_param_name": "self",
        "has_ctx": True,
        "config_params": [],
    }
    inject_and_call(func, SimpleNamespace(config={}), call_meta, ctx=sentinel)
    assert captured["ctx"] is sentinel


def test_inject_and_call_self_only() -> None:
    from neofoam.framework.operation_wrapper import inject_and_call

    def func(self: Any) -> list[str]:
        return [self.name]

    rt = SimpleNamespace(config={}, name="rt1")
    call_meta = {
        "first_param_name": "self",
        "has_ctx": False,
        "config_params": [],
    }
    assert inject_and_call(func, rt, call_meta) == ["rt1"]


def test_discover_call_metadata_extracts_first_param_ctx_and_configs() -> None:
    from neofoam.framework.operation_wrapper import discover_call_metadata

    class MyCfg(BaseConfig):
        x: int = 1

    def func(self: Any, ctx: Any, cfg: MyCfg) -> None:
        pass

    meta = discover_call_metadata(func)
    assert meta["first_param_name"] == "self"
    assert meta["has_ctx"] is True
    assert len(meta["config_params"]) == 1
    assert meta["config_params"][0]["config_type"] is MyCfg
