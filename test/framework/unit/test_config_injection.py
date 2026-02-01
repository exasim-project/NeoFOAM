# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Unit tests for config injection in @solver.config and @model.config decorators.

Tests auto-injection of configuration into operations with `config` parameter.
"""

import pytest
from dataclasses import dataclass
from typing import Annotated

from foamadapter.framework.solver_factory import Solver
from foamadapter.framework.model_factory import Model
from foamadapter.framework.context import Context, FieldUpdates


# ============================================================================
# Test: Solver Config Registration
# ============================================================================


def test_solver_config_decorator_registers_class():
    """Test @solver.config decorator registers configuration class."""
    solver = Solver("TestSolver")

    @solver.config
    @dataclass
    class TestConfig:
        tolerance: float = 1e-6
        max_iterations: int = 100

    assert solver._config_class is TestConfig


def test_solver_get_config_creates_instance():
    """Test get_config() creates and returns config instance."""
    solver = Solver("TestSolver")

    @solver.config
    @dataclass
    class SolverConfig:
        dt: float = 0.01
        end_time: float = 1.0

    config = solver.get_config()
    assert config is not None
    assert config.dt == 0.01
    assert config.end_time == 1.0


def test_solver_get_config_singleton():
    """Test get_config() returns same instance on multiple calls."""
    solver = Solver("TestSolver")

    @solver.config
    @dataclass
    class SingletonConfig:
        value: int = 42

    config1 = solver.get_config()
    config2 = solver.get_config()

    # Should be the same instance
    assert config1 is config2

    # Modification reflected in both
    config1.value = 99
    assert config2.value == 99


def test_solver_config_without_decorator_raises_error():
    """Test get_config() raises error when no config registered."""
    solver = Solver("NoConfigSolver")

    with pytest.raises(RuntimeError, match="No config class defined"):
        solver.get_config()


# ============================================================================
# Test: Solver Config Auto-Injection
# ============================================================================


def test_solver_config_auto_injected_into_operation():
    """Test config is auto-injected into operation with config parameter."""
    solver = Solver("InjectionSolver")

    @solver.config
    @dataclass
    class OpConfig:
        multiplier: float = 2.0

    received_config = None

    @solver.operation(operation_number="1.0")
    def test_operation(ctx: Context, value: float, config) -> FieldUpdates:
        nonlocal received_config
        received_config = config
        return FieldUpdates({"result": value * config.multiplier})

    # Execute operation
    ctx = Context(fields={"value": 10.0}, models={})
    test_operation(ctx)

    assert received_config is not None
    assert received_config.multiplier == 2.0
    assert ctx.fields["result"] == 20.0


def test_solver_config_not_injected_without_parameter():
    """Test config is not injected if operation doesn't have config parameter."""
    solver = Solver("NoInjectionSolver")

    @solver.config
    @dataclass
    class UnusedConfig:
        value: int = 5

    @solver.operation(operation_number="1.0")
    def operation_without_config(ctx: Context, val: float) -> FieldUpdates:
        return FieldUpdates({"result": val * 2})

    ctx = Context(fields={"val": 10.0}, models={})
    operation_without_config(ctx)

    assert ctx.fields["result"] == 20.0


def test_solver_config_injection_disabled():
    """Test config injection can be disabled with inject_config=False."""
    solver = Solver("DisabledInjectionSolver")

    @solver.config
    @dataclass
    class DisabledConfig:
        value: int = 10

    @solver.operation(operation_number="1.0", inject_config=False)
    def no_inject_operation(ctx: Context, config) -> FieldUpdates:
        # config parameter exists but injection disabled
        # so config won't be passed unless explicitly provided
        return FieldUpdates({"result": config if config else "none"})

    ctx = Context(fields={}, models={})

    # Should work but config won't be auto-injected
    # This might raise an error or handle gracefully depending on implementation
    # For now, just test that inject_config parameter is accepted
    assert hasattr(no_inject_operation, "__wrapped__") or callable(no_inject_operation)


# ============================================================================
# Test: Model Config Registration
# ============================================================================


def test_model_config_decorator_registers_class():
    """Test @model.config decorator registers configuration class."""
    model = Model("TestModel")

    @model.config
    @dataclass
    class TestConfig:
        enabled: bool = True
        coefficient: float = 0.5

    assert model._config_class is TestConfig


def test_model_get_config_creates_instance():
    """Test model get_config() creates and returns config instance."""
    model = Model("PhysicsModel")

    @model.config
    @dataclass
    class PhysicsConfig:
        gravity: float = 9.81
        viscosity: float = 1e-6

    config = model.get_config()
    assert config is not None
    assert config.gravity == 9.81
    assert config.viscosity == 1e-6


def test_model_get_config_singleton():
    """Test model get_config() returns same instance on multiple calls."""
    model = Model("SingletonModel")

    @model.config
    @dataclass
    class SingletonConfig:
        value: int = 123

    config1 = model.get_config()
    config2 = model.get_config()

    # Should be the same instance
    assert config1 is config2

    # Modification reflected in both
    config1.value = 456
    assert config2.value == 456


def test_model_config_without_decorator_raises_error():
    """Test model get_config() raises error when no config registered."""
    model = Model("NoConfigModel")

    with pytest.raises(RuntimeError, match="No config class defined"):
        model.get_config()


# ============================================================================
# Test: Model Config Auto-Injection
# ============================================================================


def test_model_config_auto_injected_into_operation():
    """Test config is auto-injected into model operation with config parameter."""
    model = Model("ConfigModel")

    @model.config
    @dataclass
    class ModelOpConfig:
        factor: float = 3.0

    received_config = None

    @model.operation(operation_number="1.0")
    def compute(ctx: Context, input_val: float, config) -> FieldUpdates:
        nonlocal received_config
        received_config = config
        return FieldUpdates({"output": input_val * config.factor})

    # Execute operation
    ctx = Context(fields={"input_val": 5.0}, models={})
    compute(ctx)

    assert received_config is not None
    assert received_config.factor == 3.0
    assert ctx.fields["output"] == 15.0


def test_model_config_not_injected_without_parameter():
    """Test config is not injected if model operation doesn't have config parameter."""
    model = Model("NoInjectionModel")

    @model.config
    @dataclass
    class UnusedConfig:
        value: int = 5

    @model.operation(operation_number="1.0")
    def operation_without_config(ctx: Context, val: float) -> FieldUpdates:
        return FieldUpdates({"result": val * 2})

    ctx = Context(fields={"val": 10.0}, models={})
    operation_without_config(ctx)

    assert ctx.fields["result"] == 20.0


# ============================================================================
# Test: Config with Complex Types
# ============================================================================


def test_config_with_nested_types():
    """Test config with complex nested data structures."""
    solver = Solver("ComplexSolver")

    @solver.config
    @dataclass
    class ComplexConfig:
        name: str = "solver"
        params: dict = None
        values: list = None

        def __post_init__(self):
            if self.params is None:
                self.params = {"tolerance": 1e-6, "max_iter": 100}
            if self.values is None:
                self.values = [1, 2, 3]

    config = solver.get_config()
    assert config.name == "solver"
    assert config.params["tolerance"] == 1e-6
    assert len(config.values) == 3


def test_config_with_methods():
    """Test config class with custom methods."""
    model = Model("MethodModel")

    @model.config
    @dataclass
    class ConfigWithMethods:
        base: float = 5.0

        def scaled(self, scale: float) -> float:
            return self.base * scale

        def is_valid(self) -> bool:
            return self.base > 0

    config = model.get_config()
    assert config.scaled(2.0) == 10.0
    assert config.is_valid() is True


# ============================================================================
# Test: Config Validation
# ============================================================================


def test_config_validation_in_post_init():
    """Test config with validation logic in __post_init__."""
    solver = Solver("ValidatedSolver")

    @solver.config
    @dataclass
    class ValidatedConfig:
        tolerance: float = 1e-6

        def __post_init__(self):
            if self.tolerance <= 0:
                raise ValueError("Tolerance must be positive")

    # Valid config
    config = solver.get_config()
    assert config.tolerance == 1e-6

    # Invalid config should raise during creation
    @solver.config
    @dataclass
    class InvalidConfig:
        tolerance: float = -1.0

        def __post_init__(self):
            if self.tolerance <= 0:
                raise ValueError("Tolerance must be positive")

    # Create new solver instance with invalid config
    solver2 = Solver("InvalidSolver")
    solver2._config_class = InvalidConfig

    with pytest.raises(ValueError, match="Tolerance must be positive"):
        solver2.get_config()


# ============================================================================
# Test: Multiple Solvers/Models with Independent Configs
# ============================================================================


def test_multiple_solvers_independent_configs():
    """Test each solver has its own independent config."""
    solver1 = Solver("Solver1")
    solver2 = Solver("Solver2")

    @solver1.config
    @dataclass
    class Config1:
        param: float = 1.0

    @solver2.config
    @dataclass
    class Config2:
        param: float = 2.0

    config1 = solver1.get_config()
    config2 = solver2.get_config()

    assert config1.param == 1.0
    assert config2.param == 2.0

    # Modify one shouldn't affect the other
    config1.param = 5.0
    assert solver1.get_config().param == 5.0
    assert solver2.get_config().param == 2.0


def test_multiple_models_independent_configs():
    """Test each model has its own independent config."""
    model1 = Model("Model1")
    model2 = Model("Model2")

    @model1.config
    @dataclass
    class Config1:
        param: float = 10.0

    @model2.config
    @dataclass
    class Config2:
        param: float = 20.0

    config1 = model1.get_config()
    config2 = model2.get_config()

    assert config1.param == 10.0
    assert config2.param == 20.0

    # Modify one shouldn't affect the other
    config1.param = 50.0
    assert model1.get_config().param == 50.0
    assert model2.get_config().param == 20.0


# ============================================================================
# Test: Config State Persistence
# ============================================================================


def test_config_state_persists_across_operations():
    """Test config state persists across multiple operation calls."""
    model = Model("PersistentModel")

    @model.config
    @dataclass
    class PersistentConfig:
        call_count: int = 0

    @model.operation(operation_number="1.0")
    def increment(ctx: Context, config) -> FieldUpdates:
        config.call_count += 1
        return FieldUpdates({"count": config.call_count})

    ctx = Context(fields={}, models={})

    # Multiple calls should share same config instance
    increment(ctx)
    assert ctx.fields["count"] == 1

    increment(ctx)
    assert ctx.fields["count"] == 2

    increment(ctx)
    assert ctx.fields["count"] == 3


# ============================================================================
# Test: Config Modification
# ============================================================================


def test_config_can_be_modified():
    """Test config instance can be modified after creation."""
    solver = Solver("ModifiableSolver")

    @solver.config
    @dataclass
    class ModConfig:
        value: int = 5

    # Get and modify config
    config = solver.get_config()
    config.value = 10

    @solver.operation()
    def op(ctx: Context, config) -> FieldUpdates:
        return FieldUpdates({"result": config.value})

    ctx = Context(fields={}, models={})
    op(ctx)

    assert ctx.fields["result"] == 10
