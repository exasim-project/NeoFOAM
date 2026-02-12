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


def test_repr_callable():
    """repr shows function name for callable dependencies."""

    def my_provider():
        return 1

    dep = Depends(my_provider)
    r = repr(dep)
    assert "my_provider" in r
    assert "time_step" in r


def test_repr_string():
    """repr shows string path for string dependencies."""
    dep = Depends("models.turbulence")
    r = repr(dep)
    assert "models.turbulence" in r


def test_repr_lambda():
    """repr handles lambda (name is '<lambda>')."""
    dep = Depends(lambda: 1)
    r = repr(dep)
    assert "<lambda>" in r


# --- __call__ ---


def test_call_callable():
    """Calling Depends with callable invokes the function."""
    dep = Depends(lambda: 42)
    assert dep() == 42


def test_call_string_raises():
    """Calling Depends with string raises TypeError."""
    dep = Depends("fields.U")
    with pytest.raises(TypeError, match="Cannot call string dependency"):
        dep()


def test_call_with_args_function():
    """Callable dependency that takes no args works with __call__."""

    def provider():
        return "result"

    dep = Depends(provider)
    assert dep() == "result"
