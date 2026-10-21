# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Unit tests for InitStep."""

from typing import Any

import pytest

from neofoam.framework.initialization.execution.executor import execute_step
from neofoam.framework.initialization.init_step import InitStep, InitStepExecutionError


def test_creation() -> None:
    """Test basic InitStep creation with all parameters."""
    lazy = InitStep(
        name="test_obj",
        depends_on=["dep1", "dep2"],
        initializer=lambda _ctx: 42,
        category="fields",
    )

    assert lazy.name == "test_obj"
    assert lazy.depends_on == ["dep1", "dep2"]
    assert lazy.initializer is not None
    assert lazy.category == "fields"


@pytest.mark.parametrize(
    "kwargs,error_match",
    [
        (
            {"name": "bad", "initializer": lambda _ctx: 1, "category": ""},
            "non-empty category",
        ),
        ({"name": "", "initializer": lambda _ctx: 42}, "non-empty name"),
        ({"name": "test", "initializer": None}, "must have an initializer"),
    ],
    ids=["empty-category", "empty-name", "none-initializer"],
)
def test_init_step_validation_raises(kwargs: dict[str, Any], error_match: str) -> None:
    """InitStep validates name, initializer, and category presence."""
    with pytest.raises(ValueError, match=error_match):
        InitStep(**kwargs)


def test_init_step_accepts_arbitrary_category_string() -> None:
    """Category is open; any non-empty string is accepted."""
    step = InitStep(
        name="custom",
        initializer=lambda _ctx: 1,
        category="turbulence",
    )
    assert step.category == "turbulence"


def test_execute_no_args() -> None:
    """Test executing InitStep whose initializer ignores the context."""
    lazy = InitStep(name="answer", initializer=lambda _ctx: 42)
    result = execute_step(lazy, {})
    assert result == 42


def test_execute_with_context() -> None:
    """Test executing InitStep with context-aware initializer."""

    def init_with_context(ctx: dict[str, int]) -> int:
        return ctx["x"] * 2

    lazy = InitStep(name="doubled", depends_on=["x"], initializer=init_with_context)

    context = {"x": 21}
    result = execute_step(lazy, context)
    assert result == 42


def test_execute_with_missing_context_key_is_wrapped() -> None:
    """Missing context key errors are wrapped with step context."""
    lazy = InitStep(name="needs_context", initializer=lambda ctx: ctx["x"])

    with pytest.raises(InitStepExecutionError, match="needs_context"):
        execute_step(lazy, {})


def test_execute_none_initializer() -> None:
    """Test that executing with None initializer raises ValueError."""
    # Bypass __post_init__ validation
    lazy = InitStep.__new__(InitStep)
    lazy.name = "broken"
    lazy.depends_on = []
    lazy.initializer = None  # type: ignore[assignment]
    lazy.category = "resource"

    with pytest.raises(ValueError, match="has no initializer"):
        execute_step(lazy, {})


def test_default_depends_on() -> None:
    """Test that depends_on defaults to empty list."""
    lazy = InitStep(name="simple", initializer=lambda _ctx: 1)
    assert lazy.depends_on == []


def test_default_category() -> None:
    """Test that category defaults to explicit 'resource'."""
    lazy = InitStep(name="uncategorized", initializer=lambda _ctx: 1)
    assert lazy.category == "resource"


def test_execute_wraps_runtime_error() -> None:
    """Non-framework exceptions are wrapped in InitStepExecutionError."""

    def exploding_init(_ctx: object) -> None:
        raise OSError("disk failure")

    step = InitStep(name="fields.T", depends_on=["mesh"], initializer=exploding_init)
    with pytest.raises(InitStepExecutionError) as exc_info:
        execute_step(step, {})

    err = exc_info.value
    assert err.step_name == "fields.T"
    assert err.depends_on == ["mesh"]
    assert "disk failure" in str(err)
    assert isinstance(err.__cause__, OSError)


def test_execute_does_not_wrap_value_error() -> None:
    """ValueError raised by initializer passes through."""
    step = InitStep(
        name="ctx_step",
        initializer=lambda _ctx: (_ for _ in ()).throw(ValueError("bad value")),
    )
    with pytest.raises(ValueError, match="bad value"):
        execute_step(step, {})


def test_execute_does_not_wrap_type_error() -> None:
    """TypeError passes through unwrapped for easier debugging of bad callables."""

    def bad_init(_ctx: object) -> None:
        raise TypeError("wrong arg type")

    step = InitStep(name="bad", initializer=bad_init)
    with pytest.raises(TypeError, match="wrong arg type"):
        execute_step(step, {})


def test_replaces_defaults_to_empty_list() -> None:
    """A step with no declared replacement supersedes nothing."""
    step = InitStep(name="mesh", initializer=lambda _ctx: None)
    assert step.replaces == []


def test_replaces_is_stored() -> None:
    """A declared replacement target is kept verbatim on the step."""
    step = InitStep(name="mesh", initializer=lambda _ctx: None, replaces=["mesh"])
    assert step.replaces == ["mesh"]
