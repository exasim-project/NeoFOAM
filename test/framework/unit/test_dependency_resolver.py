# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Unit tests for DependencyResolver.

Tests runtime dependency resolution with scope-based caching.
"""

from typing import Annotated

from foamadapter.framework.dependency_resolver import DependencyResolver
from foamadapter.framework.context import Context
from foamadapter.framework.initialization.depends import Depends


def test_resolver_creation():
    """Test DependencyResolver can be created."""
    resolver = DependencyResolver()
    assert resolver is not None
    assert hasattr(resolver, "_cache")


def test_resolve_simple_field_parameter():
    """Test resolving a simple field parameter."""
    resolver = DependencyResolver()
    ctx = Context(fields={"temperature": 300.0}, models={})

    def test_func(temperature: float):
        return temperature * 2

    kwargs = resolver.resolve_arguments(test_func, ctx)
    assert "temperature" in kwargs
    assert kwargs["temperature"] == 300.0


def test_resolve_multiple_field_parameters():
    """Test resolving multiple field parameters."""
    resolver = DependencyResolver()
    ctx = Context(
        fields={"temperature": 300.0, "pressure": 101325.0, "velocity": 10.0},
        models={},
    )

    def test_func(temperature: float, pressure: float, velocity: float):
        return temperature + pressure + velocity

    kwargs = resolver.resolve_arguments(test_func, ctx)
    assert len(kwargs) == 3
    assert kwargs["temperature"] == 300.0
    assert kwargs["pressure"] == 101325.0
    assert kwargs["velocity"] == 10.0


def test_resolve_annotated_model_parameter():
    """Test resolving Model-annotated parameters from ctx.models."""
    from foamadapter.framework.context import Model as ModelAnnotation

    resolver = DependencyResolver()

    class DummyAlgorithm:
        def solve(self):
            return True

    ctx = Context(fields={}, models={"algorithm": DummyAlgorithm()})

    def test_func(algorithm: ModelAnnotation[DummyAlgorithm]):
        return algorithm.solve()

    kwargs = resolver.resolve_arguments(test_func, ctx)
    assert "algorithm" in kwargs
    assert isinstance(kwargs["algorithm"], DummyAlgorithm)


# ============================================================================
# Test: Depends() Resolution
# ============================================================================


def test_resolve_depends_simple():
    """Test resolving Depends() with simple provider function."""
    resolver = DependencyResolver()
    ctx = Context(fields={}, models={})

    def get_config():
        return {"param": 42}

    def test_func(config: Annotated[dict, Depends(get_config)]):
        return config["param"]

    kwargs = resolver.resolve_arguments(test_func, ctx)
    assert "config" in kwargs
    assert kwargs["config"]["param"] == 42


def test_resolve_depends_with_context():
    """Test Depends() provider that needs context."""
    resolver = DependencyResolver()
    ctx = Context(fields={"value": 100}, models={})

    def get_multiplier(ctx: Context):
        return ctx.fields["value"] * 2

    def test_func(multiplier: Annotated[int, Depends(get_multiplier)]):
        return multiplier

    kwargs = resolver.resolve_arguments(test_func, ctx)
    assert "multiplier" in kwargs
    assert kwargs["multiplier"] == 200


def test_resolve_depends_nested():
    """Test nested Depends() resolution."""
    resolver = DependencyResolver()
    ctx = Context(fields={}, models={})

    def get_base():
        return 10

    def get_derived(base: Annotated[int, Depends(get_base)]):
        return base * 2

    def test_func(value: Annotated[int, Depends(get_derived)]):
        return value

    kwargs = resolver.resolve_arguments(test_func, ctx)
    assert "value" in kwargs
    assert kwargs["value"] == 20


# ============================================================================
# Test: Scope-based Caching
# ============================================================================


def test_cache_time_step_scope():
    """Test caching at time_step scope."""
    resolver = DependencyResolver()
    ctx = Context(fields={}, models={})

    call_count = {"count": 0}

    def expensive_computation():
        call_count["count"] += 1
        return call_count["count"] * 10

    def test_func(
        value: Annotated[int, Depends(expensive_computation, scope="time_step")],
    ):
        return value

    # First call
    kwargs1 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs1["value"] == 10
    assert call_count["count"] == 1

    # Second call - should use cache
    kwargs2 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs2["value"] == 10
    assert call_count["count"] == 1  # Not incremented


def test_cache_iteration_scope():
    """Test caching at iteration scope."""
    resolver = DependencyResolver()
    ctx = Context(fields={}, models={})

    call_count = {"count": 0}

    def iteration_value():
        call_count["count"] += 1
        return call_count["count"]

    def test_func(value: Annotated[int, Depends(iteration_value, scope="iteration")]):
        return value

    # First call
    kwargs1 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs1["value"] == 1

    # Second call - should use cache
    kwargs2 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs2["value"] == 1
    assert call_count["count"] == 1


def test_cache_operation_scope():
    """Test caching at operation scope."""
    resolver = DependencyResolver()
    ctx = Context(fields={}, models={})

    call_count = {"count": 0}

    def operation_value():
        call_count["count"] += 1
        return call_count["count"]

    def test_func(value: Annotated[int, Depends(operation_value, scope="operation")]):
        return value

    # First call
    kwargs1 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs1["value"] == 1

    # Second call - still cached within operation scope
    kwargs2 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs2["value"] == 1
    assert call_count["count"] == 1

    # Clear operation scope to force recompute
    resolver.clear_scope("operation")
    kwargs3 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs3["value"] == 2
    assert call_count["count"] == 2


# ============================================================================
# Test: Cache Clearing
# ============================================================================


def test_clear_iteration_scope():
    """Test clearing iteration scope cache."""
    resolver = DependencyResolver()
    ctx = Context(fields={}, models={})

    call_count = {"count": 0}

    def counter():
        call_count["count"] += 1
        return call_count["count"]

    def test_func(value: Annotated[int, Depends(counter, scope="iteration")]):
        return value

    # First call
    kwargs1 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs1["value"] == 1

    # Clear iteration cache
    resolver.clear_scope("iteration")

    # Next call should recompute
    kwargs2 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs2["value"] == 2


def test_clear_time_step_scope():
    """Test clearing time_step scope cache."""
    resolver = DependencyResolver()
    ctx = Context(fields={}, models={})

    call_count = {"count": 0}

    def counter():
        call_count["count"] += 1
        return call_count["count"]

    def test_func(value: Annotated[int, Depends(counter, scope="time_step")]):
        return value

    # First call
    kwargs1 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs1["value"] == 1

    # Clear time_step cache
    resolver.clear_scope("time_step")

    # Next call should recompute
    kwargs2 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs2["value"] == 2


def test_clear_all_scopes():
    """Test clearing all scope caches."""
    resolver = DependencyResolver()
    ctx = Context(fields={}, models={})

    count1 = {"count": 0}
    count2 = {"count": 0}

    def time_step_value():
        count1["count"] += 1
        return count1["count"]

    def iteration_value():
        count2["count"] += 1
        return count2["count"]

    def test_func(
        ts: Annotated[int, Depends(time_step_value, scope="time_step")],
        it: Annotated[int, Depends(iteration_value, scope="iteration")],
    ):
        return ts, it

    # First call
    kwargs1 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs1["ts"] == 1
    assert kwargs1["it"] == 1

    # Clear all
    resolver.clear_scope("time_step")
    resolver.clear_scope("iteration")

    # Next call should recompute both
    kwargs2 = resolver.resolve_arguments(test_func, ctx)
    assert kwargs2["ts"] == 2
    assert kwargs2["it"] == 2


# ============================================================================
# Test: Mixed Resolution
# ============================================================================


def test_mixed_fields_and_depends():
    """Test resolving mix of field parameters and Depends()."""
    resolver = DependencyResolver()
    ctx = Context(fields={"temperature": 300.0}, models={})

    def get_factor():
        return 2.0

    def test_func(temperature: float, factor: Annotated[float, Depends(get_factor)]):
        return temperature * factor

    kwargs = resolver.resolve_arguments(test_func, ctx)
    assert len(kwargs) == 2
    assert kwargs["temperature"] == 300.0
    assert kwargs["factor"] == 2.0


def test_mixed_fields_models_and_depends():
    """Test resolving mix of fields, models, and Depends()."""
    from foamadapter.framework.context import Model as ModelAnnotation

    resolver = DependencyResolver()

    class Algorithm:
        value = 5

    ctx = Context(fields={"temp": 100.0}, models={"algo": Algorithm()})

    def get_multiplier():
        return 3

    def test_func(
        temp: float,
        algo: ModelAnnotation[Algorithm],
        mult: Annotated[int, Depends(get_multiplier)],
    ):
        return temp * mult + algo.value

    kwargs = resolver.resolve_arguments(test_func, ctx)
    assert len(kwargs) == 3
    assert kwargs["temp"] == 100.0
    assert isinstance(kwargs["algo"], Algorithm)
    assert kwargs["mult"] == 3


# ============================================================================
# Test: Error Handling
# ============================================================================


def test_missing_field_behavior():
    """Test behavior with missing field - returns empty kwargs."""
    resolver = DependencyResolver()
    ctx = Context(fields={}, models={})

    def test_func(missing_field: float):
        return missing_field

    # DependencyResolver doesn't raise for simple type annotations
    # It only resolves if the field is present
    kwargs = resolver.resolve_arguments(test_func, ctx)
    # missing_field won't be in kwargs since it's not in context
    assert "missing_field" not in kwargs


def test_missing_model_behavior():
    """Test behavior with missing model - returns None from get()."""
    from foamadapter.framework.context import Model as ModelAnnotation

    resolver = DependencyResolver()
    ctx = Context(fields={}, models={})

    def test_func(missing_model: ModelAnnotation[object]):
        return missing_model

    # DependencyResolver uses .get() which returns None for missing keys
    kwargs = resolver.resolve_arguments(test_func, ctx)
    assert "missing_model" in kwargs
    assert kwargs["missing_model"] is None


# ============================================================================
# Test: Provided kwargs Override
# ============================================================================


def test_provided_kwargs_override():
    """Test that provided kwargs override resolution."""
    resolver = DependencyResolver()
    ctx = Context(fields={"temperature": 300.0}, models={})

    def test_func(temperature: float):
        return temperature

    # Provide explicit value
    kwargs = resolver.resolve_arguments(test_func, ctx, temperature=500.0)
    assert kwargs["temperature"] == 500.0  # Uses provided, not from context


def test_partial_provided_kwargs():
    """Test partial override with provided kwargs."""
    resolver = DependencyResolver()
    ctx = Context(fields={"temp": 100.0, "pressure": 200.0}, models={})

    def test_func(temp: float, pressure: float):
        return temp, pressure

    # Provide only one value
    kwargs = resolver.resolve_arguments(test_func, ctx, temp=999.0)
    assert kwargs["temp"] == 999.0  # Provided
    assert kwargs["pressure"] == 200.0  # From context
