# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for LazyInit dataclass and helper functions."""

import pytest

from foamadapter.framework.initialization.lazy_init import LazyInit
from foamadapter.framework.initialization.helpers import field, operator, lazy, model


def test_lazy_init_creation():
    """Test basic LazyInit creation."""
    lazy_init = LazyInit(name="test", depends_on=["dep1"], initializer=lambda: "result")
    assert lazy_init.name == "test"
    assert lazy_init.depends_on == ["dep1"]
    assert lazy_init.execute() == "result"


def test_lazy_init_validation_empty_name():
    """Test LazyInit validation for empty name."""
    with pytest.raises(ValueError, match="must have a non-empty name"):
        LazyInit(name="", initializer=lambda: None)


def test_lazy_init_validation_no_initializer():
    """Test LazyInit validation for missing initializer."""
    with pytest.raises(ValueError, match="must have an initializer"):
        LazyInit(name="test", initializer=None)


def test_lazy_init_execute_no_initializer():
    """Test LazyInit execute when initializer is None."""
    # This should be caught in __post_init__, but test execute anyway
    lazy_init = LazyInit.__new__(LazyInit)
    lazy_init.name = "test"
    lazy_init.depends_on = []
    lazy_init.initializer = None
    lazy_init.category = None

    with pytest.raises(ValueError, match="has no initializer function"):
        lazy_init.execute()


def test_field_helper():
    """Test field() helper function."""
    result = field("U", create=lambda: "velocity_field")

    assert result.name == "fields.U"
    assert result.category == "fields"
    assert result.depends_on == []  # Default no dependencies
    assert result.execute() == "velocity_field"


def test_field_helper_custom_dependencies():
    """Test field() helper with custom dependencies."""
    result = field("phi", create=lambda: "flux_field", depends_on=["fields.U"])

    assert result.name == "fields.phi"
    assert result.depends_on == ["fields.U"]


def test_operator_helper():
    """Test operator() helper function."""
    result = operator(
        "momentum",
        depends_on=["fields.U", "fields.p"],
        create=lambda: "momentum_equation",
    )

    assert result.name == "operators.momentum"
    assert result.category == "operators"
    assert result.depends_on == ["fields.U", "fields.p"]
    assert result.execute() == "momentum_equation"


def test_operator_helper_no_dependencies():
    """Test operator() helper with no dependencies."""
    result = operator("test_op", create=lambda: "test_operator")

    assert result.name == "operators.test_op"
    assert result.depends_on == []


def test_lazy_helper():
    """Test lazy() helper function."""
    result = lazy("mesh", create=lambda: "mesh_object")

    assert result.name == "mesh"
    assert result.category is None
    assert result.depends_on == []  # Default is empty
    assert result.execute() == "mesh_object"


def test_lazy_helper_with_dependencies():
    """Test lazy() helper with dependencies."""
    result = lazy(
        "piso_loop",
        depends_on=["operators.momentum", "operators.pressure"],
        create=lambda: "piso_loop_object",
    )

    assert result.name == "piso_loop"
    assert result.depends_on == ["operators.momentum", "operators.pressure"]


def test_model_helper():
    """Test model() helper function."""
    result = model(
        "transport",
        depends_on=["fields.U", "fields.phi"],
        create=lambda: "transport_model",
    )

    assert result.name == "models.transport"
    assert result.category == "models"
    assert result.depends_on == ["fields.U", "fields.phi"]
    assert result.execute() == "transport_model"


def test_model_helper_no_dependencies():
    """Test model() helper with no dependencies."""
    result = model("test_model", create=lambda: "test_model_object")

    assert result.name == "models.test_model"
    assert result.depends_on == []


def test_lazy_init_with_closure():
    """Test LazyInit with closure that captures variables."""
    captured_value = [None]

    def create():
        captured_value[0] = 42
        return "result"

    lazy_init = LazyInit(name="test", initializer=create)
    result = lazy_init.execute()

    assert result == "result"
    assert captured_value[0] == 42


def test_helpers_return_lazy_init():
    """Test that all helpers return LazyInit instances."""
    f = field("U", create=lambda: None)
    o = operator("momentum", create=lambda: None)
    lazy_mesh = lazy("mesh", create=lambda: None)
    m = model("transport", create=lambda: None)

    assert isinstance(f, LazyInit)
    assert isinstance(o, LazyInit)
    assert isinstance(lazy_mesh, LazyInit)
    assert isinstance(m, LazyInit)
