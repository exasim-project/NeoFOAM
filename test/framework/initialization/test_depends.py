# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Unit tests for Depends descriptor."""

import pytest

from neofoam.framework.initialization.depends import Depends


# --- Creation ---


def test_callable_dependency():
    """Depends stores a callable dependency."""

    def fn():
        return 42

    dep = Depends(fn)
    assert dep.dependency is fn


def test_string_dependency():
    """Depends stores a string path dependency."""
    dep = Depends("fields.U")
    assert dep.dependency == "fields.U"


def test_defaults():
    """Default kwargs: scope=time_step, cache=True, optional=False."""
    dep = Depends("x")
    assert dep.scope == "time_step"
    assert dep.cache is True
    assert dep.optional is False


def test_custom_kwargs():
    """Custom kwargs are stored."""
    dep = Depends("x", scope="iteration", cache=False, optional=True)
    assert dep.scope == "iteration"
    assert dep.cache is False
    assert dep.optional is True


# --- repr ---


@pytest.mark.parametrize(
    "dependency,expected",
    [
        pytest.param("models.turbulence", "models.turbulence", id="string"),
        pytest.param(lambda: 1, "<lambda>", id="lambda"),
    ],
)
def test_repr(dependency, expected):
    """repr shows dependency name/path and scope."""
    dep = Depends(dependency)
    text = repr(dep)
    assert expected in text
    assert "time_step" in text


def test_repr_callable_named_function():
    """repr shows function name for callable dependencies."""

    def my_provider():
        return 1

    dep = Depends(my_provider)
    assert "my_provider" in repr(dep)
