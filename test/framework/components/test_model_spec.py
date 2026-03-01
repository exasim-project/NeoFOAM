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
        "_resolve_call_meta": {},
        "_build_func": None,
        "_build_call_meta": {},
        "_build_operations_for": lambda rt: [],
        "_operation_collection_func": None,
        "_operations": [],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_func(n_params: int) -> Any:
    """Create a dummy function with *n_params* positional parameters."""
    if n_params == 0:
        return lambda: None
    if n_params == 1:
        return lambda a: None
    if n_params == 2:
        return lambda a, b: None
    return lambda a, b, c: None


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
    resolve_meta = {"first_param_name": "self", "has_ctx": True, "config_params": []}
    spec = _stub_spec(
        _resolve_func=lambda self, ctx: new_cfg, _resolve_call_meta=resolve_meta
    )
    rt = ModelRuntime(spec=spec, name="M_id", config={"v": 1})  # type: ignore[arg-type]
    rt.run_resolve(ctx=SimpleNamespace())  # type: ignore[arg-type]
    assert rt.config is new_cfg


def test_run_resolve_ignores_none_return_and_keeps_original() -> None:
    resolve_meta = {"first_param_name": "self", "has_ctx": True, "config_params": []}
    spec = _stub_spec(
        _resolve_func=lambda self, ctx: None, _resolve_call_meta=resolve_meta
    )
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

    build_meta = {"first_param_name": "self", "has_ctx": False, "config_params": []}
    spec = _stub_spec(_build_func=build, _build_call_meta=build_meta)
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


@pytest.mark.parametrize(
    ("decorator_name", "stored_attr", "n_params"),
    [
        ("load", "_load_func", 2),
        ("resolve", "_resolve_func", 2),
        ("build", "_build_func", 1),
    ],
    ids=["load", "resolve", "build"],
)
def test_spec_decorator_stores_function(
    decorator_name: str, stored_attr: str, n_params: int
) -> None:
    spec = ModelSpec("M")
    func = _make_func(n_params)
    getattr(spec, decorator_name)(func)
    assert getattr(spec, stored_attr) is func


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


@pytest.mark.parametrize(
    ("decorator_name", "n_params", "match"),
    [
        ("detect", 0, "@detect"),
        ("detect", 2, "@detect"),
        ("load", 1, "@load"),
        ("build", 0, "@build"),
        ("resolve", 1, "@resolve"),
    ],
    ids=["detect-0", "detect-2", "load-1", "build-0", "resolve-1"],
)
def test_decorator_rejects_wrong_param_count(
    decorator_name: str, n_params: int, match: str
) -> None:
    spec = ModelSpec("M")
    with pytest.raises(TypeError, match=match):
        getattr(spec, decorator_name)(_make_func(n_params))


@pytest.mark.parametrize(
    ("decorator_name", "n_params", "stored_attr"),
    [
        ("detect", 1, "_detect_func"),
        ("build", 1, "_build_func"),
        ("resolve", 3, "_resolve_func"),
    ],
    ids=["detect-1", "build-1", "resolve-3"],
)
def test_decorator_accepts_valid_param_count(
    decorator_name: str, n_params: int, stored_attr: str
) -> None:
    spec = ModelSpec("M")
    func = _make_func(n_params)
    getattr(spec, decorator_name)(func)
    assert getattr(spec, stored_attr) is func


# ===========================================================================
# Cycle 10 — Config metadata stored at registration
# ===========================================================================


@pytest.mark.parametrize(
    ("decorator_name", "meta_attr", "func_factory", "expected_has_ctx"),
    [
        (
            "build",
            "_build_call_meta",
            lambda MyCfg: lambda self, cfg=None: [] if cfg is None else [],
            False,
        ),
        (
            "resolve",
            "_resolve_call_meta",
            lambda MyCfg: lambda self, ctx, cfg=None: cfg,
            True,
        ),
    ],
    ids=["build", "resolve"],
)
def test_decorator_discovers_config_params(
    decorator_name: str, meta_attr: str, func_factory: Any, expected_has_ctx: bool
) -> None:
    from neofoam.io import BaseConfig

    spec = ModelSpec("M")

    class MyCfg(BaseConfig):
        x: int = 1

    # Build the real function with proper signature for each decorator
    if decorator_name == "build":

        @spec.build
        def build_fn(self: Any, cfg: MyCfg) -> list[Any]:
            return []

    else:

        @spec.resolve
        def resolve_fn(self: Any, ctx: Any, cfg: MyCfg) -> Any:
            return cfg

    meta = getattr(spec, meta_attr)
    assert len(meta["config_params"]) == 1
    assert meta["config_params"][0]["config_type"] is MyCfg
    assert meta["first_param_name"] == "self"
    assert meta["has_ctx"] is expected_has_ctx


@pytest.mark.parametrize(
    ("decorator_name", "meta_attr"),
    [
        ("build", "_build_call_meta"),
        ("resolve", "_resolve_call_meta"),
    ],
    ids=["build", "resolve"],
)
def test_config_params_empty_when_no_config(
    decorator_name: str, meta_attr: str
) -> None:
    spec = ModelSpec("M")

    if decorator_name == "build":

        @spec.build
        def build_fn(self: Any) -> list[Any]:
            return []

    else:

        @spec.resolve
        def resolve_fn(self: Any, ctx: Any) -> Any:
            return None

    assert getattr(spec, meta_attr)["config_params"] == []
