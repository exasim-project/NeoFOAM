# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Unit tests for get_call_arguments() function.

Tests parameter resolution including Annotated types, dataclasses, and Context.
"""

import pytest
from dataclasses import dataclass
from typing import Annotated, get_origin

from foamadapter.framework.context import Context
from foamadapter.framework.operations import get_call_arguments, get_function_parameters


# ============================================================================
# Test: Simple Field Parameters
# ============================================================================


def test_get_call_arguments_simple_fields():
    """Test getting simple field parameters from context."""

    def test_func(U: float, p: float) -> None:
        pass

    func_params = get_function_parameters(test_func)
    ctx = Context(fields={"U": 10.0, "p": 20.0}, models={})

    args = get_call_arguments(func_params, ctx)

    assert "U" in args
    assert "p" in args
    assert args["U"] == 10.0
    assert args["p"] == 20.0


def test_get_call_arguments_multiple_field_types():
    """Test getting fields with different types."""

    def test_func(name: str, value: float, count: int) -> None:
        pass

    func_params = get_function_parameters(test_func)
    ctx = Context(fields={"name": "test", "value": 3.14, "count": 42}, models={})

    args = get_call_arguments(func_params, ctx)

    assert args["name"] == "test"
    assert args["value"] == 3.14
    assert args["count"] == 42


# ============================================================================
# Test: Context Parameter
# ============================================================================


def test_get_call_arguments_context_parameter():
    """Test Context parameter is passed directly."""

    def test_func(ctx: Context, value: float) -> None:
        pass

    func_params = get_function_parameters(test_func)
    ctx = Context(fields={"value": 5.0}, models={})

    args = get_call_arguments(func_params, ctx)

    assert "ctx" in args
    assert args["ctx"] is ctx
    assert args["value"] == 5.0


def test_get_call_arguments_only_context():
    """Test function with only Context parameter."""

    def test_func(context: Context) -> None:
        pass

    func_params = get_function_parameters(test_func)
    ctx = Context(fields={}, models={})

    args = get_call_arguments(func_params, ctx)

    assert "context" in args
    assert args["context"] is ctx


# ============================================================================
# Test: Annotated Types (Model Access)
# ============================================================================


# Note: Annotated model access uses string like "models" not Model function
# This is tested indirectly through integration tests
# Direct test would require understanding the exact annotation pattern used


# ============================================================================
# Test: Dataclass Parameters
# ============================================================================


def test_get_call_arguments_dataclass_from_fields():
    """Test dataclass parameter constructed from context fields."""

    @dataclass
    class Config:
        tolerance: float
        max_iter: int

    def test_func(config: Config) -> None:
        pass

    func_params = get_function_parameters(test_func)
    ctx = Context(fields={"tolerance": 1e-6, "max_iter": 100}, models={})

    args = get_call_arguments(func_params, ctx)

    assert "config" in args
    assert isinstance(args["config"], Config)
    assert args["config"].tolerance == 1e-6
    assert args["config"].max_iter == 100


def test_get_call_arguments_dataclass_with_defaults():
    """Test dataclass parameter - all fields must be in context."""

    @dataclass
    class Settings:
        name: str
        enabled: bool = True
        count: int = 10

    def test_func(settings: Settings) -> None:
        pass

    func_params = get_function_parameters(test_func)
    # All fields must be provided in context
    ctx = Context(fields={"name": "test", "enabled": False, "count": 5}, models={})

    args = get_call_arguments(func_params, ctx)

    assert "settings" in args
    assert isinstance(args["settings"], Settings)
    assert args["settings"].name == "test"
    assert args["settings"].enabled is False
    assert args["settings"].count == 5


def test_get_call_arguments_nested_dataclass():
    """Test nested dataclass construction."""

    @dataclass
    class InnerConfig:
        value: float

    @dataclass
    class OuterConfig:
        inner: InnerConfig
        name: str

    def test_func(config: OuterConfig) -> None:
        pass

    func_params = get_function_parameters(test_func)

    # Note: This test depends on implementation details
    # May need adjustment based on actual nested behavior
    inner = InnerConfig(value=3.14)
    ctx = Context(fields={"inner": inner, "name": "test"}, models={})

    args = get_call_arguments(func_params, ctx)

    assert "config" in args
    assert isinstance(args["config"], OuterConfig)


# ============================================================================
# Test: Mixed Parameter Types
# ============================================================================


def test_get_call_arguments_mixed_types():
    """Test function with mix of field, Context, and dataclass parameters."""

    @dataclass
    class Params:
        alpha: float
        beta: float

    def test_func(ctx: Context, value: float, params: Params) -> None:
        pass

    func_params = get_function_parameters(test_func)
    ctx = Context(fields={"value": 10.0, "alpha": 0.5, "beta": 1.5}, models={})

    args = get_call_arguments(func_params, ctx)

    assert "ctx" in args
    assert args["ctx"] is ctx
    assert "value" in args
    assert args["value"] == 10.0
    assert "params" in args
    assert isinstance(args["params"], Params)
    assert args["params"].alpha == 0.5
    assert args["params"].beta == 1.5


# ============================================================================
# Test: get_function_parameters()
# ============================================================================


def test_get_function_parameters_simple():
    """Test extracting parameter names and types."""

    def test_func(a: int, b: float, c: str) -> None:
        pass

    params = get_function_parameters(test_func)

    assert "a" in params
    assert "b" in params
    assert "c" in params
    assert params["a"] == int
    assert params["b"] == float
    assert params["c"] == str


def test_get_function_parameters_no_annotations():
    """Test function with no type annotations."""

    def test_func(x, y, z):
        pass

    params = get_function_parameters(test_func)

    assert "x" in params
    assert "y" in params
    assert "z" in params
    # Parameters without annotations have inspect._empty as annotation


def test_get_function_parameters_with_defaults():
    """Test function with default parameter values."""

    def test_func(a: int, b: float = 3.14, c: str = "default") -> None:
        pass

    params = get_function_parameters(test_func)

    assert len(params) == 3
    assert "a" in params
    assert "b" in params
    assert "c" in params


def test_get_function_parameters_annotated_types():
    """Test extracting Annotated types."""

    class DummyType:
        pass

    def test_func(param: Annotated[DummyType, "metadata"]) -> None:
        pass

    params = get_function_parameters(test_func)

    assert "param" in params
    assert get_origin(params["param"]) is Annotated


# ============================================================================
# Test: Error Handling
# ============================================================================


def test_get_call_arguments_missing_field():
    """Test behavior when required field is missing from context."""

    def test_func(required_field: float) -> None:
        pass

    func_params = get_function_parameters(test_func)
    ctx = Context(fields={}, models={})

    # Should raise KeyError for missing field
    with pytest.raises(KeyError):
        get_call_arguments(func_params, ctx)


def test_get_call_arguments_missing_dataclass_field():
    """Test behavior when dataclass field is missing from context."""

    @dataclass
    class MissingFieldConfig:
        required: float

    def test_func(config: MissingFieldConfig) -> None:
        pass

    func_params = get_function_parameters(test_func)
    ctx = Context(fields={}, models={})

    # Should raise error when trying to construct dataclass
    with pytest.raises((KeyError, TypeError)):
        get_call_arguments(func_params, ctx)


# ============================================================================
# Test: Complex Scenarios
# ============================================================================


def test_get_call_arguments_all_types_combined():
    """Test comprehensive scenario with field, Context, and dataclass types."""

    @dataclass
    class AlgorithmConfig:
        tolerance: float
        iterations: int

    def complex_func(
        ctx: Context,
        U: float,
        config: AlgorithmConfig,
    ) -> None:
        pass

    func_params = get_function_parameters(complex_func)
    ctx = Context(fields={"U": 100.0, "tolerance": 1e-8, "iterations": 50}, models={})

    args = get_call_arguments(func_params, ctx)

    assert "ctx" in args
    assert args["ctx"] is ctx
    assert "U" in args
    assert args["U"] == 100.0
    assert "config" in args
    assert isinstance(args["config"], AlgorithmConfig)
    assert args["config"].tolerance == 1e-8
    assert args["config"].iterations == 50


def test_get_call_arguments_empty_function():
    """Test function with no parameters."""

    def test_func() -> None:
        pass

    func_params = get_function_parameters(test_func)
    ctx = Context(fields={}, models={})

    args = get_call_arguments(func_params, ctx)

    assert len(args) == 0


# ============================================================================
# Test: Type Checking
# ============================================================================


def test_parameter_types_preserved():
    """Test that parameter types are correctly identified."""

    def test_func(
        int_param: int, float_param: float, str_param: str, bool_param: bool
    ) -> None:
        pass

    params = get_function_parameters(test_func)

    assert params["int_param"] == int
    assert params["float_param"] == float
    assert params["str_param"] == str
    assert params["bool_param"] == bool
