# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Shared fixtures for initialization tests."""

import pytest

from neofoam.framework.initialization.init_step import InitStep
from neofoam.framework.initialization.helpers import InitializerBuilder
from neofoam.framework.initialization.config_context import ConfigContext


@pytest.fixture
def linear_chain():
    """A -> B -> C linear dependency chain (shuffled input order)."""
    return [
        InitStep("C", depends_on=["B"], initializer=lambda: "c"),
        InitStep("A", depends_on=[], initializer=lambda: "a"),
        InitStep("B", depends_on=["A"], initializer=lambda: "b"),
    ]


@pytest.fixture
def diamond_graph():
    """A -> (B, C) -> D diamond dependency graph."""
    return [
        InitStep("D", depends_on=["B", "C"], initializer=lambda: "d"),
        InitStep("B", depends_on=["A"], initializer=lambda: "b"),
        InitStep("C", depends_on=["A"], initializer=lambda: "c"),
        InitStep("A", depends_on=[], initializer=lambda: "a"),
    ]


@pytest.fixture
def builder():
    """Fresh InitializerBuilder instance."""
    return InitializerBuilder()


@pytest.fixture
def config():
    """Fresh ConfigContext instance."""
    return ConfigContext()


class MockCoreModel:
    """Mock model with run_build() returning InitStep objects."""

    def run_build(self):
        return [
            InitStep("test_field", initializer=lambda: "field_value"),
            InitStep("test_op", initializer=lambda: "op_value"),
        ]


class MockOptionalModel:
    """Mock optional model with run_build()."""

    def run_build(self):
        return [
            InitStep("optional_field", initializer=lambda: "opt_field"),
        ]


@pytest.fixture
def mock_core_model():
    return MockCoreModel()


@pytest.fixture
def mock_optional_model():
    return MockOptionalModel()
