# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Shared fixtures for initialization tests."""

import pytest

from neofoam.framework.initialization.init_step import InitStep
from neofoam.framework.initialization.helpers import InitializerBuilder
from neofoam.framework.initialization.config_context import ConfigContext


@pytest.fixture
def linear_chain() -> list[InitStep]:
    """A -> B -> C linear dependency chain (shuffled input order)."""
    return [
        InitStep("C", depends_on=["B"], initializer=lambda _ctx: "c"),
        InitStep("A", depends_on=[], initializer=lambda _ctx: "a"),
        InitStep("B", depends_on=["A"], initializer=lambda _ctx: "b"),
    ]


@pytest.fixture
def diamond_graph() -> list[InitStep]:
    """A -> (B, C) -> D diamond dependency graph."""
    return [
        InitStep("D", depends_on=["B", "C"], initializer=lambda _ctx: "d"),
        InitStep("B", depends_on=["A"], initializer=lambda _ctx: "b"),
        InitStep("C", depends_on=["A"], initializer=lambda _ctx: "c"),
        InitStep("A", depends_on=[], initializer=lambda _ctx: "a"),
    ]


@pytest.fixture
def builder() -> InitializerBuilder:
    """Fresh InitializerBuilder instance."""
    return InitializerBuilder()


@pytest.fixture
def config() -> ConfigContext:
    """Fresh ConfigContext instance."""
    return ConfigContext()


class MockCoreModel:
    """Mock model with run_build() returning InitStep objects."""

    def run_build(self) -> list[InitStep]:
        return [
            InitStep(
                "test_field", initializer=lambda _ctx: "field_value", category="fields"
            ),
            InitStep(
                "test_op", initializer=lambda _ctx: "op_value", category="operators"
            ),
        ]


class MockOptionalModel:
    """Mock optional model with run_build()."""

    def run_build(self) -> list[InitStep]:
        return [
            InitStep(
                "optional_field",
                initializer=lambda _ctx: "opt_field",
                category="fields",
            ),
        ]


@pytest.fixture
def mock_core_model() -> MockCoreModel:
    return MockCoreModel()


@pytest.fixture
def mock_optional_model() -> MockOptionalModel:
    return MockOptionalModel()
