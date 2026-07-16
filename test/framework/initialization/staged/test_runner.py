# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for StagedInitRunner."""

import pytest

from neofoam.framework.context import Context
from neofoam.framework.initialization.config_context import ConfigContext
from neofoam.framework.initialization.helpers import field, lazy, model
from neofoam.framework.initialization.staged.runner import StagedInitRunner
from neofoam.framework.initialization.staged.spec import (
    LoadResult,
    StagedInitSpec,
)


# --- Construction & defaults ---


def test_runner_defaults():
    spec = StagedInitSpec.build("TestSolver").finalize()
    runner = StagedInitRunner(spec)

    assert runner.name == "TestSolver"
    assert runner.argv == []
    assert runner.core_models == []
    assert runner.optional_models == []
    assert runner.spec is spec


def test_runner_accepts_argv():
    spec = StagedInitSpec.build("S").finalize()
    runner = StagedInitRunner(spec, argv=["--case", "/tmp"])

    assert runner.argv == ["--case", "/tmp"]


# --- run() error cases ---


def test_run_no_load_raises():
    builder = StagedInitSpec.build("X")

    @builder.build
    def build(core, opt):
        _ = (core, opt)
        return []

    runner = StagedInitRunner(builder.finalize())
    with pytest.raises(RuntimeError, match="No @X.load defined"):
        runner.run()


def test_run_no_build_raises():
    builder = StagedInitSpec.build("X")

    @builder.load
    def load():
        return LoadResult(core_models=[], optional_models=[])

    runner = StagedInitRunner(builder.finalize())
    with pytest.raises(RuntimeError, match="No @X.build defined"):
        runner.run()


# --- run() full pipeline ---


def _make_full_runner() -> tuple[StagedInitRunner, dict]:
    builder = StagedInitSpec.build("Test")
    observed: dict = {"order": [], "resolve_cfg": None}

    class CoreModel:
        name = "core1"

    class OptionalModel:
        name = "opt1"

    @builder.load
    def load_stage():
        observed["order"].append("load")
        return LoadResult(core_models=[CoreModel()], optional_models=[OptionalModel()])

    @builder.resolve
    def resolve_stage(cfg):
        observed["order"].append("resolve")
        observed["resolve_cfg"] = cfg

    @builder.build
    def build_stage(core, opt):
        observed["order"].append("build")
        assert len(core) == 1
        assert len(opt) == 1
        return [
            lazy("mesh", create=lambda _ctx: "mesh_obj"),
            field("U", depends_on=["mesh"], create=lambda _ctx: "velocity"),
            model("algo", create=lambda _ctx: "algorithm"),
        ]

    return StagedInitRunner(builder.finalize()), observed


def test_run_full_pipeline():
    runner, observed = _make_full_runner()
    ctx = runner.run()

    assert isinstance(ctx, Context)
    assert ctx.mesh == "mesh_obj"
    assert ctx.fields == {"U": "velocity"}
    assert "algo" in ctx.models
    assert observed["order"] == ["load", "resolve", "build"]
    assert isinstance(observed["resolve_cfg"], ConfigContext)
    assert len(runner.core_models) == 1
    assert len(runner.optional_models) == 1
    assert runner.core_models[0].name == "core1"
    assert runner.optional_models[0].name == "opt1"


def test_run_without_resolve():
    builder = StagedInitSpec.build("X")

    @builder.load
    def load():
        return LoadResult(core_models=[], optional_models=[])

    @builder.build
    def build(core, opt):
        _ = (core, opt)
        return [lazy("mesh", create=lambda _ctx: "m")]

    runner = StagedInitRunner(builder.finalize())
    ctx = runner.run()
    assert ctx.mesh == "m"


@pytest.mark.parametrize(
    "models,expected_key",
    [
        ([type("FakeAlgo", (), {})()], "fakealgo"),
        ([type("NamedModel", (), {"name": "transport"})()], "transport"),
    ],
    ids=["class-name-key", "model-name-key"],
)
def test_run_wires_models_into_config(models, expected_key):
    builder = StagedInitSpec.build("X")
    captured: dict = {}

    @builder.load
    def load():
        return LoadResult(core_models=models, optional_models=[])

    @builder.resolve
    def resolve(cfg):
        captured["registered"] = cfg.get(expected_key)

    @builder.build
    def build(core, opt):
        _ = (core, opt)
        return [lazy("mesh", create=lambda _ctx: "m")]

    StagedInitRunner(builder.finalize()).run()
    assert captured["registered"] is models[0]


def test_run_duplicate_model_registration_key_raises():
    builder = StagedInitSpec.build("X")

    class M1:
        name = "dup"

    class M2:
        name = "dup"

    @builder.load
    def load():
        return LoadResult(core_models=[M1(), M2()], optional_models=[])

    @builder.build
    def build(core, opt):
        _ = (core, opt)
        return [lazy("mesh", create=lambda _ctx: "m")]

    runner = StagedInitRunner(builder.finalize())
    with pytest.raises(ValueError, match="Duplicate model registration key"):
        runner.run()


# --- run_load / run_resolve / run_build standalone ---


def test_run_load_returns_load_result():
    builder = StagedInitSpec.build("X")

    @builder.load
    def load():
        return LoadResult(core_models=["c"], optional_models=["o"])

    runner = StagedInitRunner(builder.finalize())
    lr = runner.run_load()
    assert isinstance(lr, LoadResult)
    assert lr.core_models == ["c"]
    assert runner.core_models == ["c"]
    assert runner.optional_models == ["o"]


def test_run_load_no_func_raises():
    spec = StagedInitSpec.build("X").finalize()
    runner = StagedInitRunner(spec)
    with pytest.raises(RuntimeError, match="No @X.load defined"):
        runner.run_load()


def test_run_build_returns_lazy_inits():
    builder = StagedInitSpec.build("X")

    @builder.build
    def build(core, opt):
        _ = (core, opt)
        return [lazy("mesh", create=lambda _ctx: "mesh_obj")]

    runner = StagedInitRunner(builder.finalize())
    runner.core_models = ["c1", "c2"]
    runner.optional_models = ["o1"]

    result = runner.run_build()
    assert len(result) == 1
    assert result[0].name == "mesh"


def test_run_build_passes_models():
    builder = StagedInitSpec.build("X")
    captured: dict = {}

    @builder.build
    def build(core, opt):
        captured["core"] = core
        captured["opt"] = opt
        return [lazy("mesh", create=lambda _ctx: "mesh_obj")]

    runner = StagedInitRunner(builder.finalize())
    runner.core_models = ["c1", "c2"]
    runner.optional_models = ["o1"]

    runner.run_build()
    assert captured["core"] == ["c1", "c2"]
    assert captured["opt"] == ["o1"]


def test_run_build_no_func_raises():
    spec = StagedInitSpec.build("X").finalize()
    runner = StagedInitRunner(spec)
    with pytest.raises(RuntimeError, match="No @X.build defined"):
        runner.run_build()


def test_run_resolve_passes_config():
    builder = StagedInitSpec.build("X")
    config = ConfigContext()
    captured: dict = {}

    @builder.resolve
    def resolve(cfg):
        captured["cfg"] = cfg

    runner = StagedInitRunner(builder.finalize())
    runner.run_resolve(config)
    assert captured["cfg"] is config


def test_run_resolve_without_resolve_is_noop():
    spec_no_resolve = StagedInitSpec.build("Y").finalize()
    StagedInitRunner(spec_no_resolve).run_resolve(ConfigContext())
