# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for StagedInitSpec, StagedInitSpecBuilder, and LoadResult."""

import dataclasses

import pytest

from neofoam.framework.initialization.staged.spec import (
    LoadResult,
    StagedInitSpec,
    StagedInitSpecBuilder,
)


# --- LoadResult ---


def test_load_result_all_models():
    lr = LoadResult(core_models=["c1", "c2"], optional_models=["o1"])
    assert lr.all_models == ["c1", "c2", "o1"]


def test_load_result_all_models_empty():
    lr = LoadResult(core_models=[], optional_models=[])
    assert lr.all_models == []


def test_load_result_validate_empty_returns_no_errors():
    """With no configs, validate() runs the IO validator and returns an empty list."""
    lr = LoadResult(core_models=[], optional_models=[])
    assert lr.validate() == []


def test_load_result_configs_empty():
    lr = LoadResult(core_models=["plain_obj"], optional_models=[])
    assert lr.configs == []


def test_load_result_configs_collects():
    class ModelWithConfigs:
        configs = ["cfg1", "cfg2"]

    lr = LoadResult(core_models=[ModelWithConfigs()], optional_models=[])
    assert lr.configs == ["cfg1", "cfg2"]


# --- StagedInitSpec / StagedInitSpecBuilder ---


def test_spec_build_returns_builder():
    builder = StagedInitSpec.build("TestSolver")
    assert isinstance(builder, StagedInitSpecBuilder)


def test_builder_load_decorator_records_callback():
    builder = StagedInitSpec.build("X")

    @builder.load
    def my_load():
        return LoadResult(core_models=[], optional_models=[])

    spec = builder.finalize()
    assert spec.load_fn is my_load


def test_builder_resolve_decorator_records_callback():
    builder = StagedInitSpec.build("X")

    @builder.resolve
    def my_resolve(cfg):
        _ = cfg

    spec = builder.finalize()
    assert spec.resolve_fn is my_resolve


def test_builder_build_decorator_records_callback():
    builder = StagedInitSpec.build("X")

    @builder.build
    def my_build(core, opt):
        _ = (core, opt)
        return []

    spec = builder.finalize()
    assert spec.build_fn is my_build


def test_spec_is_frozen():
    spec = StagedInitSpec.build("X").finalize()

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


def test_spec_defaults_are_none():
    spec = StagedInitSpec.build("Y").finalize()

    assert spec.name == "Y"
    assert spec.load_fn is None
    assert spec.resolve_fn is None
    assert spec.build_fn is None
