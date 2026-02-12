# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Unit tests for InitStep."""

import pytest

from neofoam.framework.initialization.init_step import InitStep


def test_creation():
    """Test basic InitStep creation with all parameters."""
    lazy = InitStep(
        name="test_obj",
        depends_on=["dep1", "dep2"],
        initializer=lambda: 42,
        category="test_category",
    )

    assert lazy.name == "test_obj"
    assert lazy.depends_on == ["dep1", "dep2"]
    assert lazy.initializer is not None
    assert lazy.category == "test_category"


def test_empty_name_raises():
    """Test that empty name raises ValueError."""
    with pytest.raises(ValueError, match="non-empty name"):
        InitStep(name="", initializer=lambda: 42)


def test_none_initializer_raises():
    """Test that None initializer raises ValueError."""
    with pytest.raises(ValueError, match="must have an initializer"):
        InitStep(name="test", initializer=None)


def test_execute_no_args():
    """Test executing InitStep with no-argument initializer."""
    lazy = InitStep(name="answer", initializer=lambda: 42)
    result = lazy.execute()
    assert result == 42


def test_execute_with_context():
    """Test executing InitStep with context-aware initializer."""

    def init_with_context(ctx):
        return ctx["x"] * 2

    lazy = InitStep(name="doubled", depends_on=["x"], initializer=init_with_context)

    context = {"x": 21}
    result = lazy.execute(context)
    assert result == 42


def test_execute_with_context_no_context_provided():
    """Test that context-aware initializer raises when context is None."""
    lazy = InitStep(name="needs_context", initializer=lambda ctx: ctx["x"])

    with pytest.raises(ValueError, match="requires context"):
        lazy.execute(context=None)


def test_execute_none_initializer():
    """Test that executing with None initializer raises ValueError."""
    # Bypass __post_init__ validation
    lazy = InitStep.__new__(InitStep)
    lazy.name = "broken"
    lazy.depends_on = []
    lazy.initializer = None
    lazy.category = None

    with pytest.raises(ValueError, match="has no initializer"):
        lazy.execute()


def test_default_depends_on():
    """Test that depends_on defaults to empty list."""
    lazy = InitStep(name="simple", initializer=lambda: 1)
    assert lazy.depends_on == []


def test_default_category():
    """Test that category defaults to None."""
    lazy = InitStep(name="uncategorized", initializer=lambda: 1)
    assert lazy.category is None
