# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Unit tests for StagedInit, LoadResult, and ValidationError."""

import pytest

from neofoam.framework.initialization.staged_init import (
    StagedInit,
    LoadResult,
    ValidationError,
)
from neofoam.framework.initialization.config_context import ConfigContext
from neofoam.framework.initialization.helpers import field, lazy, model
from neofoam.framework.context import Context


# --- ValidationError ---


def test_validation_error_creation():
    """ValidationError stores field, message, and default severity."""
    err = ValidationError(field="pressure", message="out of range")
    assert err.field == "pressure"
    assert err.message == "out of range"
    assert err.severity == "error"


def test_validation_error_warning():
    """ValidationError accepts custom severity."""
    err = ValidationError(field="nu", message="low", severity="warning")
    assert err.severity == "warning"


# --- LoadResult ---


def test_load_result_all_models():
    """all_models concatenates core + optional."""
    lr = LoadResult(core_models=["c1", "c2"], optional_models=["o1"])
    assert lr.all_models == ["c1", "c2", "o1"]


def test_load_result_all_models_empty():
    """all_models works with empty lists."""
    lr = LoadResult(core_models=[], optional_models=[])
    assert lr.all_models == []


def test_load_result_validate_raises():
    """validate() raises NotImplementedError (IO-coupled stub)."""
    lr = LoadResult(core_models=[], optional_models=[])
    with pytest.raises(NotImplementedError):
        lr.validate()


def test_load_result_configs_empty():
    """configs returns empty list when models have no .configs."""
    lr = LoadResult(core_models=["plain_obj"], optional_models=[])
    assert lr.configs == []


def test_load_result_configs_collects():
    """configs collects from models that have a .configs attribute."""

    class ModelWithConfigs:
        configs = ["cfg1", "cfg2"]

    lr = LoadResult(core_models=[ModelWithConfigs()], optional_models=[])
    assert lr.configs == ["cfg1", "cfg2"]


# --- StagedInit creation ---


def test_creation():
    """StagedInit stores name and defaults."""
    init = StagedInit("TestSolver")
    assert init.name == "TestSolver"
    assert init.argv == []
    assert init._load_func is None
    assert init._resolve_func is None
    assert init._build_func is None


def test_creation_with_argv():
    """StagedInit accepts argv."""
    init = StagedInit("S", argv=["--case", "/tmp"])
    assert init.argv == ["--case", "/tmp"]


# --- Decorator registration ---


def test_load_decorator_registers():
    """@init.load registers the function."""
    init = StagedInit("X")

    @init.load
    def my_load():
        return LoadResult(core_models=[], optional_models=[])

    assert init._load_func is my_load


def test_resolve_decorator_registers():
    """@init.resolve registers the function."""
    init = StagedInit("X")

    @init.resolve
    def my_resolve(core, opt, cfg):
        pass

    assert init._resolve_func is my_resolve


def test_build_decorator_registers():
    """@init.build registers the function."""
    init = StagedInit("X")

    @init.build
    def my_build(core, opt):
        return []

    assert init._build_func is my_build


# --- State properties ---


def test_state_properties():
    """core_models / optional_models / configs proxy to SolverState."""
    init = StagedInit("X")
    assert init.core_models == []
    assert init.optional_models == []
    assert init.configs == {}

    init.core_models = ["a"]
    init.optional_models = ["b"]
    init.configs = {"k": "v"}

    assert init.state.core_models == ["a"]
    assert init.state.optional_models == ["b"]
    assert init.state.configs == {"k": "v"}


# --- run() error cases ---


def test_run_no_load_raises():
    """run() raises if no @init.load is defined."""
    init = StagedInit("X")

    @init.build
    def build(core, opt):
        return []

    with pytest.raises(RuntimeError, match="No @X.load defined"):
        init.run()


def test_run_no_build_raises():
    """run() raises if no @init.build is defined."""
    init = StagedInit("X")

    @init.load
    def load():
        return LoadResult(core_models=[], optional_models=[])

    with pytest.raises(RuntimeError, match="No @X.build defined"):
        init.run()


# --- run() full pipeline ---


def _make_full_init():
    """Helper: create a StagedInit with all 3 stages wired up."""
    init = StagedInit("Test")
    order = []

    @init.load
    def load_stage():
        order.append("load")
        return LoadResult(core_models=["core1"], optional_models=["opt1"])

    @init.resolve
    def resolve_stage(core, opt, cfg):
        order.append("resolve")

    @init.build
    def build_stage(core, opt):
        order.append("build")
        return [
            lazy("mesh", create=lambda: "mesh_obj"),
            field("U", depends_on=["mesh"], create=lambda ctx: "velocity"),
            model("algo", create=lambda: "algorithm"),
        ]

    return init, order


def test_run_full():
    """run() executes load → resolve → build and returns Context."""
    init, _ = _make_full_init()
    ctx = init.run()

    assert isinstance(ctx, Context)
    assert ctx.mesh == "mesh_obj"
    assert ctx.fields == {"U": "velocity"}
    assert "algo" in ctx.models


def test_run_stage_order():
    """run() executes stages in order: load → resolve → build."""
    init, order = _make_full_init()
    init.run()
    assert order == ["load", "resolve", "build"]


def test_run_populates_state():
    """run() populates core_models and optional_models from LoadResult."""
    init, _ = _make_full_init()
    init.run()
    assert init.core_models == ["core1"]
    assert init.optional_models == ["opt1"]


def test_run_without_resolve():
    """run() works without @init.resolve (resolve is optional)."""
    init = StagedInit("X")

    @init.load
    def load():
        return LoadResult(core_models=[], optional_models=[])

    @init.build
    def build(core, opt):
        return [lazy("mesh", create=lambda: "m")]

    ctx = init.run()
    assert ctx.mesh == "m"


# --- run_load / run_resolve / run_build standalone ---


def test_run_load_returns_load_result():
    """run_load() returns LoadResult and populates state."""
    init = StagedInit("X")

    @init.load
    def load():
        return LoadResult(core_models=["c"], optional_models=["o"])

    lr = init.run_load()
    assert isinstance(lr, LoadResult)
    assert lr.core_models == ["c"]
    assert init.core_models == ["c"]
    assert init.optional_models == ["o"]


def test_run_load_no_func_raises():
    """run_load() raises if no @init.load is defined."""
    init = StagedInit("X")
    with pytest.raises(RuntimeError, match="No @X.load defined"):
        init.run_load()


def test_run_build_returns_lazy_inits():
    """run_build() returns list[LazyInit] using current state."""
    init = StagedInit("X")
    init.core_models = ["core"]
    init.optional_models = ["opt"]

    @init.build
    def build(core, opt):
        return [lazy("mesh", create=lambda: f"mesh_from_{len(core)}_core")]

    result = init.run_build()
    assert len(result) == 1
    assert result[0].name == "mesh"


def test_run_build_passes_models():
    """run_build() passes core_models and optional_models to build func."""
    init = StagedInit("X")
    init.core_models = ["c1", "c2"]
    init.optional_models = ["o1"]
    captured = {}

    @init.build
    def build(core, opt):
        captured["core"] = core
        captured["opt"] = opt
        return []

    init.run_build()
    assert captured["core"] == ["c1", "c2"]
    assert captured["opt"] == ["o1"]


def test_run_build_no_func_raises():
    """run_build() raises if no @init.build is defined."""
    init = StagedInit("X")
    with pytest.raises(RuntimeError, match="No @X.build defined"):
        init.run_build()


def test_run_resolve_3arg():
    """run_resolve() passes core_models, optional_models, config."""
    init = StagedInit("X")
    init.core_models = ["c"]
    init.optional_models = ["o"]
    captured = {}

    @init.resolve
    def resolve(core, opt, cfg):
        captured["core"] = core
        captured["opt"] = opt
        captured["cfg"] = cfg

    config = ConfigContext()
    init.run_resolve(config)
    assert captured["core"] == ["c"]
    assert captured["opt"] == ["o"]
    assert isinstance(captured["cfg"], ConfigContext)


def test_run_resolve_1arg():
    """run_resolve() supports legacy 1-arg resolve(config)."""
    init = StagedInit("X")
    captured = {}

    @init.resolve
    def resolve(cfg):
        captured["cfg"] = cfg

    config = ConfigContext()
    init.run_resolve(config)
    assert isinstance(captured["cfg"], ConfigContext)


def test_run_resolve_no_func_noop():
    """run_resolve() is a no-op if no @init.resolve is defined."""
    init = StagedInit("X")
    config = ConfigContext()
    # Should not raise
    init.run_resolve(config)


def test_run_build_0arg():
    """run_build() supports 0-arg build functions (no model params)."""
    init = StagedInit("X")

    @init.build
    def build():
        return [lazy("mesh", create=lambda: "m")]

    result = init.run_build()
    assert len(result) == 1
    assert result[0].name == "mesh"


def test_run_build_bad_arity_raises():
    """run_build() raises RuntimeError for unsupported arity (e.g. 1)."""
    init = StagedInit("X")

    @init.build
    def build(only_one):
        return []

    with pytest.raises(RuntimeError, match="Expected 0 or 2 parameters"):
        init.run_build()


def test_run_wires_models_into_config():
    """run() registers LoadResult models into ConfigContext for resolve."""
    init = StagedInit("X")
    captured_cfg = {}

    class FakeAlgo:
        pass

    @init.load
    def load():
        return LoadResult(core_models=[FakeAlgo()], optional_models=[])

    @init.resolve
    def resolve(core, opt, cfg):
        # cfg should have our model registered under its lowered class name
        captured_cfg["algo"] = cfg.get("fakealgo")

    @init.build
    def build(core, opt):
        return [lazy("mesh", create=lambda: "m")]

    init.run()
    assert isinstance(captured_cfg["algo"], FakeAlgo)


def test_run_with_0arg_build():
    """Full run() with a 0-arg build function."""
    init = StagedInit("X")

    @init.load
    def load():
        return LoadResult(core_models=[], optional_models=[])

    @init.build
    def build():
        return [lazy("mesh", create=lambda: "m")]

    ctx = init.run()
    assert ctx.mesh == "m"
