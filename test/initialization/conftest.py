# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Shared fixtures for initialization tests."""

import pytest

from neofoam.framework.initialization.lazy_init import LazyInit
from neofoam.framework.initialization.helpers import InitializerBuilder
from neofoam.framework.initialization.config_context import ConfigContext


@pytest.fixture
def linear_chain():
    """A -> B -> C linear dependency chain (shuffled input order)."""
    return [
        LazyInit("C", depends_on=["B"], initializer=lambda: "c"),
        LazyInit("A", depends_on=[], initializer=lambda: "a"),
        LazyInit("B", depends_on=["A"], initializer=lambda: "b"),
    ]


@pytest.fixture
def diamond_graph():
    """A -> (B, C) -> D diamond dependency graph."""
    return [
        LazyInit("D", depends_on=["B", "C"], initializer=lambda: "d"),
        LazyInit("B", depends_on=["A"], initializer=lambda: "b"),
        LazyInit("C", depends_on=["A"], initializer=lambda: "c"),
        LazyInit("A", depends_on=[], initializer=lambda: "a"),
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
    """Mock model with run_build() returning LazyInit objects."""

    def run_build(self):
        return [
            LazyInit("test_field", initializer=lambda: "field_value"),
            LazyInit("test_op", initializer=lambda: "op_value"),
        ]


class MockOptionalModel:
    """Mock optional model with run_build()."""

    def run_build(self):
        return [
            LazyInit("optional_field", initializer=lambda: "opt_field"),
        ]


@pytest.fixture
def mock_core_model():
    return MockCoreModel()


@pytest.fixture
def mock_optional_model():
    return MockOptionalModel()
