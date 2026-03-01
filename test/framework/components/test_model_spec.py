# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Unit tests for ModelSpec / ModelRuntime.

Follows strict TDD: each test exercises one piece of behaviour.
"""

from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

from neofoam.framework.model import ModelRuntime, ModelSpec, Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_spec(**kwargs: Any) -> SimpleNamespace:
    """Return a minimal spec-like namespace for testing ModelRuntime in isolation."""
    defaults: dict[str, Any] = {
        "_resolve_func": None,
        "_resolve_config_params": [],
        "_build_func": None,
        "_build_config_params": [],
        "_build_operations_for": lambda rt: [],
        "_operation_collection_func": None,
        "_operations": [],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ===========================================================================
# Cycle 1 — ModelRuntime: construction and basic properties
# ===========================================================================


def test_model_runtime_stores_name_and_config() -> None:
    spec = _stub_spec()
    rt = ModelRuntime(spec=spec, name="HeatSource_zoneA", config={"power": 100})  # type: ignore[arg-type]
    assert rt.name == "HeatSource_zoneA"
    assert rt.config == {"power": 100}
    assert rt.spec is spec  # type: ignore[comparison-overlap]  # stub is SimpleNamespace


def test_run_resolve_with_no_resolve_func_leaves_config_unchanged() -> None:
    spec = _stub_spec(_resolve_func=None)
    rt = ModelRuntime(spec=spec, name="M_id", config={"v": 1})  # type: ignore[arg-type]
    rt.run_resolve(ctx=SimpleNamespace())  # type: ignore[arg-type]
    assert rt.config == {"v": 1}


def test_run_resolve_calls_spec_func_and_stores_returned_config() -> None:
    new_cfg = {"v": 99}
    spec = _stub_spec(_resolve_func=lambda self, ctx: new_cfg)
    rt = ModelRuntime(spec=spec, name="M_id", config={"v": 1})  # type: ignore[arg-type]
    rt.run_resolve(ctx=SimpleNamespace())  # type: ignore[arg-type]
    assert rt.config is new_cfg


def test_run_resolve_ignores_none_return_and_keeps_original() -> None:
    spec = _stub_spec(_resolve_func=lambda self, ctx: None)
    original = {"v": 1}
    rt = ModelRuntime(spec=spec, name="M_id", config=original)  # type: ignore[arg-type]
    rt.run_resolve(ctx=SimpleNamespace())  # type: ignore[arg-type]
    assert rt.config is original


def test_run_build_returns_empty_list_when_no_build_func() -> None:
    spec = _stub_spec(_build_func=None)
    rt = ModelRuntime(spec=spec, name="M_id", config={})  # type: ignore[arg-type]
    assert rt.run_build() == []


def test_run_build_calls_spec_func_with_config_and_returns_result() -> None:
    captured: dict[str, Any] = {}

    def build(self: Any) -> list[Any]:
        captured["config"] = self.config
        return ["step_a", "step_b"]

    spec = _stub_spec(_build_func=build)
    rt = ModelRuntime(spec=spec, name="M_id", config={"x": 7})  # type: ignore[arg-type]
    result = rt.run_build()
    assert captured["config"] == {"x": 7}
    assert result == ["step_a", "step_b"]  # type: ignore[comparison-overlap]


def test_operations_delegates_to_spec_build_operations_for() -> None:
    from neofoam.framework.operations import Operations

    sentinel = object()
    spec = _stub_spec(_build_operations_for=lambda rt: [sentinel])
    rt = ModelRuntime(spec=spec, name="M_id", config={})  # type: ignore[arg-type]
    result = rt.operations
    assert isinstance(result, Operations)
    assert list(result) == [sentinel]


# ===========================================================================
# Cycle 2 — ModelSpec: decorator API
# ===========================================================================


def test_model_spec_stores_name() -> None:
    spec = ModelSpec("HeatSource")
    assert spec.name == "HeatSource"


def test_model_factory_returns_model_spec() -> None:
    spec = Model("MyModel")
    assert isinstance(spec, ModelSpec)
    assert spec.name == "MyModel"


def test_spec_load_decorator_stores_function() -> None:
    spec = ModelSpec("M")

    @spec.load
    def load(case_dir: Any, entry: Any) -> Any:
        return {"v": 1}

    assert spec._load_func is load


def test_spec_resolve_decorator_stores_function() -> None:
    spec = ModelSpec("M")

    @spec.resolve
    def resolve(self: Any, ctx: Any) -> Any:
        return self.config

    assert spec._resolve_func is resolve


def test_spec_build_decorator_stores_function() -> None:
    spec = ModelSpec("M")

    @spec.build
    def build(self: Any) -> list[Any]:
        return []

    assert spec._build_func is build


def test_spec_detect_defaults_to_true() -> None:
    spec = ModelSpec("M")
    result = spec.run_detect(Path("."))
    assert result.detected is True
    assert result.instance_ids == []


def test_spec_detect_decorator_stores_and_runs_function() -> None:
    spec = ModelSpec("M")

    @spec.detect
    def detect(_case_dir: Path) -> bool:
        return False

    result = spec.run_detect(Path("."))
    assert result.detected is False
    assert result.instance_ids == []


# ===========================================================================
# Cycle 3 — ModelSpec.instantiate creates a ModelRuntime
# ===========================================================================


def test_instantiate_without_load_or_config_raises() -> None:
    """ModelSpec.instantiate() must raise ValueError if no @load or @config is registered."""
    spec = ModelSpec("M")
    with pytest.raises(ValueError):
        spec.instantiate(Path("."))


def test_instantiate_calls_load_and_returns_runtime() -> None:
    spec = ModelSpec("HeatSource")

    @spec.load
    def load(case_dir: Any, entry: Any) -> Any:
        return {"power": 42, "name": entry["name"]}

    entry = {"type": "HeatSource", "name": "zoneA"}
    rt = spec.instantiate(case_dir=Path("."), entry=entry)

    assert isinstance(rt, ModelRuntime)
    assert rt.name == "zoneA"
    assert rt.config == {"power": 42, "name": "zoneA"}
    assert rt.spec is spec


def test_instantiate_different_entries_produce_independent_runtimes() -> None:
    spec = ModelSpec("HeatSource")

    @spec.load
    def load(case_dir: Any, entry: Any) -> Any:
        return {"name": entry["name"]}

    rt_a = spec.instantiate(Path("."), entry={"type": "HeatSource", "name": "zoneA"})
    rt_b = spec.instantiate(Path("."), entry={"type": "HeatSource", "name": "zoneB"})

    assert rt_a.name == "zoneA"
    assert rt_b.name == "zoneB"
    assert rt_a.config is not rt_b.config


# ===========================================================================
# Cycle 4 — State isolation: N runtimes from the same spec never share state
# ===========================================================================


def test_multiple_runtimes_have_independent_configs() -> None:
    """Resolving one runtime must not affect another's config."""
    spec = ModelSpec("M")

    @spec.load
    def load(case_dir: Any, entry: Any) -> Any:
        return {"v": 0}

    rt_a = spec.instantiate(Path("."), entry={"type": "M", "name": "a"})
    rt_b = spec.instantiate(Path("."), entry={"type": "M", "name": "b"})

    # Mutate rt_a's config via resolve
    rt_a.run_resolve(
        ctx=SimpleNamespace(  # type: ignore[arg-type]
            _resolve_func=None,
            all=lambda: {},
        )
    )
    rt_a.config["v"] = 99

    assert rt_b.config["v"] == 0, "rt_b config was corrupted by rt_a mutation"


def test_run_build_results_are_independent_per_runtime() -> None:
    """run_build() on two runtimes with different configs produces different results."""
    spec = ModelSpec("M")

    @spec.load
    def load(case_dir: Any, entry: Any) -> Any:
        return {"value": float(entry["name"])}

    @spec.build
    def build(self: Any) -> list[Any]:
        return [self.config["value"]]  # simplistic: return the value as the "step"

    rt_42 = spec.instantiate(Path("."), entry={"type": "M", "name": "42"})
    rt_99 = spec.instantiate(Path("."), entry={"type": "M", "name": "99"})

    assert rt_42.run_build() == [42.0]  # type: ignore[comparison-overlap]
    assert rt_99.run_build() == [99.0]  # type: ignore[comparison-overlap]


# ===========================================================================
# Cycle 5 — Config injection error cases
# ===========================================================================


def testfind_config_by_type_raises_on_missing_type() -> None:
    """find_config_by_type raises ValueError when no match exists."""
    from neofoam.framework.operation_wrapper import find_config_by_type
    from neofoam.io import BaseConfig

    class MyConfig(BaseConfig):
        x: int = 1

    class OtherConfig(BaseConfig):
        y: int = 2

    cfg = MyConfig()
    with pytest.raises(ValueError, match="OtherConfig"):
        find_config_by_type(cfg, OtherConfig)


def testfind_config_by_type_finds_direct_match() -> None:
    """find_config_by_type returns the config when it directly matches."""
    from neofoam.framework.operation_wrapper import find_config_by_type
    from neofoam.io import BaseConfig

    class MyConfig(BaseConfig):
        x: int = 5

    cfg = MyConfig()
    assert find_config_by_type(cfg, MyConfig) is cfg


def testfind_config_by_type_searches_namespace() -> None:
    """find_config_by_type locates a config nested inside a SimpleNamespace."""
    from neofoam.framework.operation_wrapper import find_config_by_type
    from neofoam.io import BaseConfig

    class StepConfig(BaseConfig):
        factor: float = 0.01

    class MainConfig(BaseConfig):
        prop: float = 1.0

    ns = SimpleNamespace(step=StepConfig(), main=MainConfig())
    assert isinstance(find_config_by_type(ns, StepConfig), StepConfig)
    assert isinstance(find_config_by_type(ns, MainConfig), MainConfig)


def testfind_config_by_type_raises_when_not_in_namespace() -> None:
    """find_config_by_type raises when config type missing from namespace."""
    from neofoam.framework.operation_wrapper import find_config_by_type
    from neofoam.io import BaseConfig

    class Missing(BaseConfig):
        pass

    ns = SimpleNamespace()
    with pytest.raises(ValueError, match="Missing"):
        find_config_by_type(ns, Missing)


# ===========================================================================
# Cycle 6 — ModelRuntime.configs property
# ===========================================================================


def test_runtime_configs_returns_single_base_config() -> None:
    """When config is a BaseConfig, configs property returns a one-element list."""
    from neofoam.io import BaseConfig

    class MyConfig(BaseConfig):
        v: int = 1

    spec = _stub_spec()
    cfg = MyConfig()
    rt = ModelRuntime(spec=spec, name="rt", config=cfg)  # type: ignore[arg-type]

    assert rt.configs == [cfg]


def test_runtime_configs_returns_all_from_namespace() -> None:
    """When config is a SimpleNamespace of BaseConfigs, all are returned."""
    from neofoam.io import BaseConfig

    class ConfigA(BaseConfig):
        a: int = 1

    class ConfigB(BaseConfig):
        b: int = 2

    spec = _stub_spec()
    ca, cb = ConfigA(), ConfigB()
    rt = ModelRuntime(spec=spec, name="rt", config=SimpleNamespace(a=ca, b=cb))  # type: ignore[arg-type]

    assert sorted(rt.configs, key=id) == sorted([ca, cb], key=id)


def test_runtime_configs_empty_for_non_config() -> None:
    """When config is not a BaseConfig or namespace, configs returns []."""
    spec = _stub_spec()
    rt = ModelRuntime(spec=spec, name="rt", config={"plain": "dict"})  # type: ignore[arg-type]
    assert rt.configs == []


# ===========================================================================
# Cycle 7 — @config decorator
# ===========================================================================


def test_config_decorator_stores_config_class() -> None:
    """@spec.config registers the config class on the spec."""
    from neofoam.io import BaseConfig

    spec = ModelSpec("M")

    @spec.config
    class MyCfg(BaseConfig):
        x: int = 1

    assert spec._config_class is MyCfg


def test_instantiate_with_entry_uses_config_class() -> None:
    """When @config is registered and entry is provided, config is auto-constructed."""
    from neofoam.io import BaseConfig

    spec = ModelSpec("TestModel")

    @spec.config
    class Cfg(BaseConfig):
        scale: float = 0.0
        offset: float = 0.0

    entry = {"type": "TestModel", "name": "inst_a", "scale": 1.5, "offset": 0.1}
    rt = spec.instantiate(case_dir=Path("."), entry=entry)

    assert isinstance(rt, ModelRuntime)
    assert rt.name == "inst_a"
    assert rt.config.scale == 1.5
    assert rt.config.offset == 0.1


def test_instantiate_with_entry_name_equals_spec_name() -> None:
    """When entry name equals spec name, runtime name should be the spec name."""
    from neofoam.io import BaseConfig

    spec = ModelSpec("Solo")

    @spec.config
    class Cfg(BaseConfig):
        x: int = 0

    entry = {"type": "Solo", "name": "Solo", "x": 42}
    rt = spec.instantiate(case_dir=Path("."), entry=entry)
    assert rt.name == "Solo"
    assert rt.config.x == 42


def test_instantiate_without_entry_or_load_raises() -> None:
    """instantiate() with no @load and no entry raises ValueError."""
    from neofoam.io import BaseConfig

    spec = ModelSpec("M")

    @spec.config
    class Cfg(BaseConfig):
        x: int = 0

    with pytest.raises(ValueError):
        spec.instantiate(case_dir=Path("."))


def test_instantiate_with_load_override_uses_load() -> None:
    """When @load is registered, it takes priority over @config even with entry."""
    from neofoam.io import BaseConfig

    spec = ModelSpec("M")

    @spec.config
    class Cfg(BaseConfig):
        x: int = 0

    @spec.load
    def load(case_dir: Any, entry: Any) -> dict[str, Any]:
        return {"custom": True}

    entry = {"type": "M", "name": "custom_inst", "x": 99}
    rt = spec.instantiate(case_dir=Path("."), entry=entry)
    assert rt.config == {"custom": True}
    assert rt.name == "custom_inst"


# ===========================================================================
# Cycle 8 — Decorator param-count validation
# ===========================================================================


def test_detect_requires_exactly_one_param() -> None:
    spec = ModelSpec("M")
    with pytest.raises(TypeError, match="@detect"):

        @spec.detect
        def bad_detect() -> bool:
            return True


def test_detect_rejects_extra_params() -> None:
    spec = ModelSpec("M")
    with pytest.raises(TypeError, match="@detect"):

        @spec.detect
        def bad_detect(a: Any, b: Any) -> bool:
            return True


def test_detect_accepts_one_param() -> None:
    spec = ModelSpec("M")

    @spec.detect
    def ok(_case_dir: Path) -> bool:
        return True

    assert spec._detect_func is ok


def test_load_requires_exactly_two_params() -> None:
    spec = ModelSpec("M")
    with pytest.raises(TypeError, match="@load"):

        @spec.load
        def bad_load(case_dir: Any) -> Any:
            return {}


def test_build_accepts_self_only() -> None:
    spec = ModelSpec("M")

    @spec.build
    def build(self: Any) -> list[Any]:
        return []

    assert spec._build_func is build


def test_build_rejects_zero_params() -> None:
    spec = ModelSpec("M")
    with pytest.raises(TypeError, match="@build"):

        @spec.build
        def bad() -> list[Any]:
            return []


def test_resolve_accepts_three_params() -> None:
    spec = ModelSpec("M")

    @spec.resolve
    def resolve(self: Any, ctx: Any, cfg: Any) -> Any:
        return cfg

    assert spec._resolve_func is resolve


def test_resolve_rejects_one_param() -> None:
    spec = ModelSpec("M")
    with pytest.raises(TypeError, match="@resolve"):

        @spec.resolve
        def bad(self: Any) -> Any:
            return None


# ===========================================================================
# Cycle 9 — inject_and_call helper
# ===========================================================================


def test_inject_and_call_binds_self_and_injects_config() -> None:
    from neofoam.framework.operation_wrapper import inject_and_call
    from neofoam.io import BaseConfig

    class MyCfg(BaseConfig):
        val: int = 42

    captured: dict[str, Any] = {}

    def func(self: Any, cfg: MyCfg) -> list[Any]:
        captured["self"] = self
        captured["cfg"] = cfg
        return ["step"]

    runtime = SimpleNamespace(config=MyCfg(val=99))
    config_params = [{"param_name": "cfg", "config_type": MyCfg}]
    result = inject_and_call(func, runtime, config_params)
    assert captured["self"] is runtime
    assert captured["cfg"].val == 99
    assert result == ["step"]


def test_inject_and_call_passes_ctx() -> None:
    from neofoam.framework.operation_wrapper import inject_and_call

    captured: dict[str, Any] = {}

    def func(self: Any, ctx: Any) -> None:
        captured["ctx"] = ctx

    sentinel = object()
    inject_and_call(func, SimpleNamespace(config={}), [], ctx=sentinel)
    assert captured["ctx"] is sentinel


def test_inject_and_call_self_only() -> None:
    from neofoam.framework.operation_wrapper import inject_and_call

    def func(self: Any) -> list[str]:
        return [self.name]

    rt = SimpleNamespace(config={}, name="rt1")
    assert inject_and_call(func, rt, []) == ["rt1"]


# ===========================================================================
# Cycle 10 — Config metadata stored at registration
# ===========================================================================


def test_build_discovers_config_params() -> None:
    from neofoam.io import BaseConfig

    spec = ModelSpec("M")

    class MyCfg(BaseConfig):
        x: int = 1

    @spec.build
    def build(self: Any, cfg: MyCfg) -> list[Any]:
        return []

    assert len(spec._build_config_params) == 1
    assert spec._build_config_params[0]["config_type"] is MyCfg


def test_resolve_discovers_config_params() -> None:
    from neofoam.io import BaseConfig

    spec = ModelSpec("M")

    class MyCfg(BaseConfig):
        x: int = 1

    @spec.resolve
    def resolve(self: Any, ctx: Any, cfg: MyCfg) -> Any:
        return cfg

    assert len(spec._resolve_config_params) == 1
    assert spec._resolve_config_params[0]["config_type"] is MyCfg


def test_build_config_params_empty_when_no_config() -> None:
    spec = ModelSpec("M")

    @spec.build
    def build(self: Any) -> list[Any]:
        return []

    assert spec._build_config_params == []


def test_resolve_config_params_empty_when_no_config() -> None:
    spec = ModelSpec("M")

    @spec.resolve
    def resolve(self: Any, ctx: Any) -> Any:
        return None

    assert spec._resolve_config_params == []
