# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Unit tests for StagedInit, LoadResult, and ValidationError."""

from typing import Any

import pytest

from neofoam.framework.context import Context
from neofoam.framework.initialization.config_context import ConfigContext
from neofoam.framework.initialization.helpers import field, lazy, model
from neofoam.framework.initialization.init_step import InitStep
from neofoam.framework.initialization.staged_init import (
    LoadResult,
    StagedInit,
    ValidationError,
)


# --- ValidationError ---


def test_validation_error_creation() -> None:
    """ValidationError stores field, message, and default severity."""
    err = ValidationError(field="pressure", message="out of range")
    assert err.field == "pressure"
    assert err.message == "out of range"
    assert err.severity == "error"


def test_validation_error_warning() -> None:
    """ValidationError accepts custom severity."""
    err = ValidationError(field="nu", message="low", severity="warning")
    assert err.severity == "warning"


# --- LoadResult ---


def test_load_result_all_models() -> None:
    """all_models concatenates core + optional."""
    lr = LoadResult(core_models=["c1", "c2"], optional_models=["o1"])
    assert lr.all_models == ["c1", "c2", "o1"]


def test_load_result_all_models_empty() -> None:
    """all_models works with empty lists."""
    lr = LoadResult(core_models=[], optional_models=[])
    assert lr.all_models == []


def test_load_result_validate_empty() -> None:
    """validate() returns empty list for empty load results."""
    lr = LoadResult(core_models=[], optional_models=[])
    assert lr.validate() == []


def test_load_result_configs_empty() -> None:
    """configs returns empty list when models have no .configs."""
    lr = LoadResult(core_models=["plain_obj"], optional_models=[])
    assert lr.configs == []


def test_load_result_configs_collects() -> None:
    """configs collects from models that have a .configs attribute."""

    class ModelWithConfigs:
        configs = ["cfg1", "cfg2"]

    lr = LoadResult(core_models=[ModelWithConfigs()], optional_models=[])
    assert lr.configs == ["cfg1", "cfg2"]


# --- StagedInit creation & decorators ---


def test_creation() -> None:
    """StagedInit stores name and defaults."""
    init = StagedInit("TestSolver")
    assert init.name == "TestSolver"
    assert init.argv == []
    assert init._hooks.load is None
    assert init._hooks.resolve is None
    assert init._hooks.build is None
    assert init.core_models == []
    assert init.optional_models == []


def test_creation_with_argv() -> None:
    """StagedInit accepts argv."""
    init = StagedInit("S", argv=["--case", "/tmp"])
    assert init.argv == ["--case", "/tmp"]


def test_load_decorator_registers() -> None:
    """@init.load registers the function."""
    init = StagedInit("X")

    @init.load
    def my_load() -> LoadResult:
        return LoadResult(core_models=[], optional_models=[])

    assert init._hooks.load is my_load


def test_resolve_decorator_registers() -> None:
    """@init.resolve registers the function."""
    init = StagedInit("X")

    @init.resolve
    def my_resolve(cfg: Any) -> None:
        _ = cfg

    assert init._hooks.resolve is my_resolve


def test_build_decorator_registers() -> None:
    """@init.build registers the function."""
    init = StagedInit("X")

    @init.build
    def my_build(core: list[Any], opt: list[Any]) -> list[InitStep]:
        _ = (core, opt)
        return []

    assert init._hooks.build is my_build


# --- run() error cases ---


def test_run_no_load_raises() -> None:
    """run() raises if no @init.load is defined."""
    init = StagedInit("X")

    @init.build
    def build(core: list[Any], opt: list[Any]) -> list[InitStep]:
        _ = (core, opt)
        return []

    with pytest.raises(RuntimeError, match="No @X.load defined"):
        init.run()


def test_run_no_build_raises() -> None:
    """run() raises if no @init.build is defined."""
    init = StagedInit("X")

    @init.load
    def load() -> LoadResult:
        return LoadResult(core_models=[], optional_models=[])

    with pytest.raises(RuntimeError, match="No @X.build defined"):
        init.run()


# --- run() full pipeline ---


def _make_full_init() -> tuple[StagedInit, dict[str, Any]]:
    """Helper: create a StagedInit with all 3 stages wired up."""
    init = StagedInit("Test")
    observed: dict[str, Any] = {"order": [], "resolve_cfg": None}

    class CoreModel:
        name = "core1"

    class OptionalModel:
        name = "opt1"

    @init.load
    def load_stage() -> LoadResult:
        observed["order"].append("load")
        return LoadResult(core_models=[CoreModel()], optional_models=[OptionalModel()])

    @init.resolve
    def resolve_stage(cfg: Any) -> None:
        observed["order"].append("resolve")
        observed["resolve_cfg"] = cfg

    @init.build
    def build_stage(core: list[Any], opt: list[Any]) -> list[InitStep]:
        observed["order"].append("build")
        assert len(core) == 1
        assert len(opt) == 1
        return [
            init("mesh", create=lambda _ctx: "mesh_obj"),
            field("U", depends_on=["mesh"], create=lambda _ctx: "velocity"),
            model("algo", create=lambda _ctx: "algorithm"),
        ]

    return staged_init, observed


def test_run_full_pipeline() -> None:
    """run() executes stages in order and populates context + runtime models."""
    staged_init, observed = _make_full_init()
    ctx = staged_init.run()

    assert isinstance(ctx, Context)
    assert ctx.mesh == "mesh_obj"
    assert ctx.fields == {"U": "velocity"}
    assert "algo" in ctx.models
    assert observed["order"] == ["load", "resolve", "build"]
    assert isinstance(observed["resolve_cfg"], ConfigContext)
    assert len(staged_init.core_models) == 1
    assert len(staged_init.optional_models) == 1
    assert getattr(staged_init.core_models[0], "name") == "core1"
    assert getattr(staged_init.optional_models[0], "name") == "opt1"


def test_run_without_resolve() -> None:
    """run() works without @init.resolve (resolve is optional)."""
    staged_init = StagedInit("X")

    @init.load
    def load() -> LoadResult:
        return LoadResult(core_models=[], optional_models=[])

    @init.build
    def build(core: list[Any], opt: list[Any]) -> list[InitStep]:
        _ = (core, opt)
        return [init("mesh", create=lambda _ctx: "m")]

    ctx = staged_init.run()
    assert ctx.mesh == "m"


@pytest.mark.parametrize(
    "models,expected_key",
    [
        ([type("FakeAlgo", (), {})()], "fakealgo"),
        ([type("NamedModel", (), {"name": "transport"})()], "transport"),
    ],
    ids=["class-name-key", "model-name-key"],
)
def test_run_wires_models_into_config(models: list[Any], expected_key: str) -> None:
    """run() registers loaded models into ConfigContext with expected key."""
    staged_init = StagedInit("X")
    captured_cfg = {}

    @init.load
    def load() -> LoadResult:
        return LoadResult(core_models=models, optional_models=[])

    @init.resolve
    def resolve(cfg: Any) -> None:
        captured_cfg["registered"] = cfg.get(expected_key)

    @init.build
    def build(core: list[Any], opt: list[Any]) -> list[InitStep]:
        _ = (core, opt)
        return [init("mesh", create=lambda _ctx: "m")]

    staged_init.run()
    assert captured_cfg["registered"] is models[0]


def test_run_duplicate_model_registration_key_raises() -> None:
    """run() raises when two models resolve to the same registration key."""
    staged_init = StagedInit("X")

    class M1:
        name = "dup"

    class M2:
        name = "dup"

    @init.load
    def load() -> LoadResult:
        return LoadResult(core_models=[M1(), M2()], optional_models=[])

    @init.build
    def build(core: list[Any], opt: list[Any]) -> list[InitStep]:
        _ = (core, opt)
        return [init("mesh", create=lambda _ctx: "m")]

    with pytest.raises(ValueError, match="Duplicate model registration key"):
        staged_init.run()


# --- run_load / run_resolve / run_build standalone ---


def test_run_load_returns_load_result() -> None:
    """run_load() returns LoadResult and populates state."""
    staged_init = StagedInit("X")

    @init.load
    def load() -> LoadResult:
        return LoadResult(core_models=["c"], optional_models=["o"])

    lr = staged_init.run_load()
    assert isinstance(lr, LoadResult)
    assert lr.core_models == ["c"]
    assert staged_init.core_models == ["c"]
    assert staged_init.optional_models == ["o"]


def test_run_load_no_func_raises() -> None:
    """run_load() raises if no @init.load is defined."""
    init = StagedInit("X")
    with pytest.raises(RuntimeError, match="No @X.load defined"):
        staged_init.run_load()


def test_run_build_returns_lazy_inits_and_passes_models() -> None:
    """run_build() passes models and returns list[InitStep]."""
    staged_init = StagedInit("X")
    staged_init.core_models = ["c1", "c2"]
    staged_init.optional_models = ["o1"]
    captured = {}

    @init.build
    def build(core: list[Any], opt: list[Any]) -> list[InitStep]:
        captured["core"] = core
        captured["opt"] = opt
        return [init("mesh", create=lambda _ctx: f"mesh_from_{len(core)}_core")]

    result = staged_init.run_build()
    assert len(result) == 1
    assert result[0].name == "mesh"
    assert captured["core"] == ["c1", "c2"]
    assert captured["opt"] == ["o1"]


def test_run_build_no_func_raises() -> None:
    """run_build() raises if no @init.build is defined."""
    init = StagedInit("X")
    with pytest.raises(RuntimeError, match="No @X.build defined"):
        staged_init.run_build()


def test_run_resolve_passes_config_and_is_optional_noop() -> None:
    """run_resolve() passes ConfigContext and noops when unresolved."""
    staged_init = StagedInit("X")
    config = ConfigContext()
    captured = {}

    @init.resolve
    def resolve(cfg: Any) -> None:
        captured["cfg"] = cfg

    staged_init.run_resolve(config)
    assert captured["cfg"] is config

    staged_init_no_resolve = StagedInit("Y")
    staged_init_no_resolve.run_resolve(ConfigContext())
