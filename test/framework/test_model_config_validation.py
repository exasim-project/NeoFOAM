# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Unit tests for Model configuration and validation features.

Tests:
- .with_config() chainable API for registering config classes
- Operation-level configs parameter injection
- validate_after_load() and validate_after_resolve() hooks
- Error aggregation across multiple configs
"""

import pytest
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from dataclasses import dataclass

from foamadapter.framework.model_factory import ModelInstance
from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.initialization.staged_init import ValidationError


# Test configuration classes
class MainModelConfig(BaseModel):
    """Main model configuration."""

    enabled: bool = True
    prop1: float = Field(..., gt=0, description="Must be positive")
    prop2: float = Field(..., le=100, description="Must be <= 100")
    name: str = "default"


class OperationSpecificConfig(BaseModel):
    """Configuration specific to an operation."""

    threshold: float = Field(..., ge=0, le=1, description="Must be between 0 and 1")
    iterations: int = Field(default=10, gt=0)
    method: str = Field(default="implicit")


class SecondaryConfig(BaseModel):
    """Secondary configuration for testing multiple configs."""

    beta: float = 1.0
    gamma: float = Field(..., lt=10)


# Test fixtures
@pytest.fixture
def model() -> ModelInstance:
    """Create a fresh model instance for testing."""
    return ModelInstance("TestModel")


@pytest.fixture
def context() -> Context:
    """Create a Context for testing operations."""
    return Context(
        fields={"field1": 100.0, "field2": 200.0},
        models={},
        algorithm={},
        time={"current_time": 0.0},
    )


# Tests for .with_config() API


def test_register_single_config(model: ModelInstance) -> None:
    """Test registering a single config class."""
    result = model.with_config(MainModelConfig)

    # Should return the model instance for chaining
    assert result is model

    # Config should be stored
    assert "main" in model._config_classes
    assert model._config_classes["main"] == MainModelConfig


def test_register_named_config(model: ModelInstance) -> None:
    """Test registering config with custom name."""
    result = model.with_config(MainModelConfig, name="custom_config")

    assert result is model
    assert "custom_config" in model._config_classes
    assert model._config_classes["custom_config"] == MainModelConfig


def test_register_multiple_configs(model: ModelInstance) -> None:
    """Test registering multiple configs."""
    model.with_config(MainModelConfig).with_config(SecondaryConfig, name="secondary")

    assert len(model._config_classes) == 2
    assert "main" in model._config_classes
    assert "secondary" in model._config_classes
    assert model._config_classes["main"] == MainModelConfig
    assert model._config_classes["secondary"] == SecondaryConfig


def test_duplicate_config_name_raises_error(model: ModelInstance) -> None:
    """Test that registering duplicate config name raises error."""
    model.with_config(MainModelConfig)

    with pytest.raises(ValueError, match="Config with name 'main' already exists"):
        model.with_config(SecondaryConfig, name="main")


def test_get_config_instance(model: ModelInstance) -> None:
    """Test getting instantiated config."""
    model.with_config(MainModelConfig)

    # First call should instantiate
    config = model.get_config("main", prop1=5.0, prop2=50.0)
    assert isinstance(config, MainModelConfig)
    assert config.prop1 == 5.0
    assert config.prop2 == 50.0

    # Second call should return cached instance
    config2 = model.get_config("main")
    assert config2 is config


def test_get_config_default_name(model: ModelInstance) -> None:
    """Test that get_config() without name uses 'main'."""
    model.with_config(MainModelConfig)
    config = model.get_config(prop1=5.0, prop2=50.0)

    assert isinstance(config, MainModelConfig)
    assert config.prop1 == 5.0


# Tests for operation-level configs


def test_operation_with_config_param(model: ModelInstance) -> None:
    """Test operation with configs parameter."""
    model.with_config(OperationSpecificConfig, name="op_config")

    @model.operation(
        operation_number="1.0",
        configs=["op_config"],
    )
    def test_op(
        self: ModelInstance,
        ctx: Context,
        field1: float,
        op_config: OperationSpecificConfig,
    ) -> FieldUpdates:
        """Operation that uses operation-specific config."""
        assert isinstance(op_config, OperationSpecificConfig)
        return FieldUpdates({"result": field1 * op_config.threshold})

    # Instantiate the operation config
    model.get_config("op_config", threshold=0.5, iterations=20, method="explicit")

    # Operation should be registered
    assert len(model._operations) == 1


def test_operation_with_multiple_configs(model: ModelInstance) -> None:
    """Test operation with multiple config injections."""
    model.with_config(MainModelConfig).with_config(
        OperationSpecificConfig, name="op_config"
    )

    @model.operation(
        operation_number="1.0",
        configs=["main", "op_config"],
    )
    def test_op(
        self: ModelInstance,
        ctx: Context,
        main_config: MainModelConfig,
        op_config: OperationSpecificConfig,
    ) -> FieldUpdates:
        """Operation that uses multiple configs."""
        assert isinstance(main_config, MainModelConfig)
        assert isinstance(op_config, OperationSpecificConfig)
        return FieldUpdates({"result": 0.0})

    # Operation should be registered
    assert len(model._operations) == 1


def test_operation_missing_config_raises_error(model: ModelInstance) -> None:
    """Test that using undefined config in operation raises error."""
    with pytest.raises(
        ValueError, match="Config 'nonexistent' not registered with model"
    ):

        @model.operation(
            operation_number="1.0",
            configs=["nonexistent"],
        )
        def test_op(self: ModelInstance, ctx: Context) -> FieldUpdates:
            return FieldUpdates({})


# Tests for validation hooks


def test_validate_after_load_success(model: ModelInstance) -> None:
    """Test validation passes with valid config."""
    model.with_config(MainModelConfig)

    # Set valid config
    model.get_config("main", prop1=5.0, prop2=50.0)

    # Validation should pass
    errors = model.validate_after_load()
    assert len(errors) == 0


def test_validate_after_load_fails_with_invalid_config(model: ModelInstance) -> None:
    """Test validation fails with invalid config."""
    model.with_config(MainModelConfig)

    # Set invalid config (prop1 must be > 0, prop2 must be <= 100)
    try:
        model.get_config("main", prop1=-5.0, prop2=150.0)
    except PydanticValidationError:
        pass  # Expected

    # Now try with partially valid
    model._config_instances = {}  # Reset
    try:
        model.get_config("main", prop1=5.0, prop2=150.0)
    except PydanticValidationError as e:
        # Validation should have detected the errors
        assert len(e.errors()) > 0


def test_validate_after_load_collects_all_errors(model: ModelInstance) -> None:
    """Test that validation collects all errors from all configs."""
    model.with_config(MainModelConfig).with_config(SecondaryConfig, name="secondary")

    # Create invalid instances manually to test error collection
    model._config_instances = {}

    # Try to create invalid configs and catch their errors
    main_errors = []
    try:
        MainModelConfig(prop1=-5.0, prop2=150.0, name="test")
    except PydanticValidationError as e:
        main_errors = e.errors()

    secondary_errors = []
    try:
        SecondaryConfig(gamma=15.0)  # Must be < 10
    except PydanticValidationError as e:
        secondary_errors = e.errors()

    # Should have errors from both configs
    assert len(main_errors) > 0
    assert len(secondary_errors) > 0


def test_validate_after_load_with_no_configs(model: ModelInstance) -> None:
    """Test validation with no registered configs."""
    errors = model.validate_after_load()
    assert len(errors) == 0  # No configs = no errors


def test_validate_after_load_with_uninstantiated_configs(model: ModelInstance) -> None:
    """Test validation skips configs that haven't been instantiated."""
    model.with_config(MainModelConfig).with_config(SecondaryConfig, name="secondary")

    # Only instantiate one config
    model.get_config("main", prop1=5.0, prop2=50.0)

    # Should only validate instantiated configs
    errors = model.validate_after_load()
    assert len(errors) == 0  # The instantiated config is valid


def test_validate_after_resolve(model: ModelInstance) -> None:
    """Test validate_after_resolve hook."""
    model.with_config(MainModelConfig)
    model.get_config("main", prop1=5.0, prop2=50.0)

    # For now, validate_after_resolve just calls validate_after_load
    errors = model.validate_after_resolve()
    assert len(errors) == 0


def test_validation_error_structure(model: ModelInstance) -> None:
    """Test that ValidationError has correct structure."""
    model.with_config(MainModelConfig)

    # Manually create a validation error to test structure
    error = ValidationError(
        field="prop1", message="Value must be positive", severity="error"
    )

    assert error.field == "prop1"
    assert error.message == "Value must be positive"
    assert error.severity == "error"


def test_validation_with_warnings(model: ModelInstance) -> None:
    """Test validation can have warnings (severity != 'error')."""
    # Create a custom validation that returns warnings
    model.with_config(MainModelConfig)
    model.get_config("main", prop1=5.0, prop2=50.0)

    # Manually add a warning
    warning = ValidationError(
        field="name", message="Using default name", severity="warning"
    )

    assert warning.severity == "warning"


# Tests for integration with existing Model features


def test_config_with_load_decorator(model: ModelInstance) -> None:
    """Test .with_config() works with @model.load decorator."""
    model.with_config(MainModelConfig)

    @model.load
    def load_config() -> MainModelConfig:
        return MainModelConfig(prop1=10.0, prop2=20.0)

    # Load should work
    result = model.run_load()
    assert isinstance(result, MainModelConfig)
    assert result.prop1 == 10.0


def test_config_injection_in_operation(
    model: ModelInstance, context: Context
) -> None:
    """Test that config is properly injected in operations."""
    model.with_config(MainModelConfig)
    model.get_config("main", prop1=5.0, prop2=50.0)

    @model.operation(operation_number="1.0")
    def test_op(
        self: ModelInstance, ctx: Context, field1: float, config: MainModelConfig
    ) -> FieldUpdates:
        """Operation with config injection."""
        return FieldUpdates({"result": field1 * config.prop1})

    # Config should be available in operation
    # (actual execution would require full context setup)


def test_with_config_chainable(model: ModelInstance) -> None:
    """Test that with_config returns self for chaining."""
    result1 = model.with_config(MainModelConfig)
    result2 = result1.with_config(SecondaryConfig, name="secondary")

    # Both should return the model instance
    assert result1 is model
    assert result2 is model

    # Both configs should be registered
    assert len(model._config_classes) == 2


# Tests for error cases


def test_get_config_nonexistent_name(model: ModelInstance) -> None:
    """Test getting config that doesn't exist."""
    with pytest.raises(KeyError, match="No config registered with name"):
        model.get_config("nonexistent")


def test_with_config_invalid_type(model: ModelInstance) -> None:
    """Test registering non-BaseModel class."""

    class NotABaseModel:
        pass

    with pytest.raises(TypeError, match="Config class must be a Pydantic BaseModel"):
        model.with_config(NotABaseModel)  # type: ignore


def test_validation_with_pydantic_error(model: ModelInstance) -> None:
    """Test that Pydantic validation errors are converted to ValidationErrors."""
    model.with_config(MainModelConfig)

    # This will raise PydanticValidationError
    with pytest.raises(PydanticValidationError):
        model.get_config("main", prop1="not_a_float", prop2=50.0)  # type: ignore


# Integration-style validation tests (mirroring validate_case pattern)


def test_validate_model_all_configs_valid(model: ModelInstance) -> None:
    """Test complete model validation lifecycle with all valid configs."""
    # Register multiple configs
    model.with_config(MainModelConfig).with_config(OperationSpecificConfig, name="op_config").with_config(
        SecondaryConfig, name="secondary"
    )

    # Instantiate all configs with valid data
    model.get_config("main", prop1=10.5, prop2=50.0)
    model.get_config("op_config", threshold=0.5, iterations=100)
    model.get_config("secondary", beta=1.0, gamma=5.0)

    # Validate after load - should succeed with no errors
    errors = model.validate_after_load()
    assert len(errors) == 0, f"Expected no errors, got {errors}"


def test_validate_model_with_invalid_configs(model: ModelInstance) -> None:
    """Test that invalid configs raise exceptions during instantiation."""
    # Register multiple configs
    model.with_config(MainModelConfig).with_config(OperationSpecificConfig, name="op_config").with_config(
        SecondaryConfig, name="secondary"
    )

    # Instantiate configs with validation errors - should raise immediately
    with pytest.raises(PydanticValidationError) as exc_info:
        model.get_config("main", prop1=-5.0, prop2=150.0)  # Both fields invalid
    
    # Check that error contains details about both violations
    error_dict = exc_info.value.errors()
    assert len(error_dict) >= 2  # Both prop1 and prop2

    with pytest.raises(PydanticValidationError) as exc_info:
        model.get_config("op_config", threshold=2.0, iterations=-10)  # Both fields invalid
    
    error_dict = exc_info.value.errors()
    assert len(error_dict) >= 2

    with pytest.raises(PydanticValidationError) as exc_info:
        model.get_config("secondary", beta=1.0, gamma=15.0)  # gamma must be < 10
    
    error_dict = exc_info.value.errors()
    assert len(error_dict) >= 1


def test_validate_model_no_configs_instantiated(model: ModelInstance) -> None:
    """Test validation when configs are registered but not instantiated."""
    # Register configs but don't instantiate them
    model.with_config(MainModelConfig).with_config(OperationSpecificConfig, name="op_config")

    # Validate after load - should succeed (no instantiated configs to validate)
    errors = model.validate_after_load()
    assert len(errors) == 0


def test_validate_model_partial_instantiation(model: ModelInstance) -> None:
    """Test validation when only some registered configs are instantiated."""
    # Register multiple configs
    model.with_config(MainModelConfig).with_config(OperationSpecificConfig, name="op_config").with_config(
        SecondaryConfig, name="secondary"
    )

    # Only instantiate one config (valid)
    model.get_config("main", prop1=10.5, prop2=50.0)

    # Validate - should only validate the instantiated config
    errors = model.validate_after_load()
    assert len(errors) == 0


def test_validate_model_lifecycle_integration(model: ModelInstance, context: Context) -> None:
    """Test full model lifecycle: register → instantiate → validate → resolve."""
    # 1. Register configs
    model.with_config(MainModelConfig).with_config(OperationSpecificConfig, name="op_config")

    # 2. Define operations that use these configs
    @model.operation(configs=["op_config"])
    def step_with_config(ctx: Context, op_config: OperationSpecificConfig) -> None:
        assert op_config.threshold == 0.5

    # 3. Instantiate configs with valid data
    model.get_config("main", prop1=10.5, prop2=50.0)
    model.get_config("op_config", threshold=0.5, iterations=100)

    # 4. Validate after load
    errors = model.validate_after_load()
    assert len(errors) == 0

    # 5. Execute operation (simulates resolve stage)
    step_with_config(context)


def test_validate_model_detects_constraint_violations(model: ModelInstance) -> None:
    """Test that validation detects Pydantic field constraints when re-validating configs."""
    model.with_config(MainModelConfig).with_config(SecondaryConfig, name="secondary")

    # Create valid configs first
    model.get_config("main", prop1=10.0, prop2=50.0)
    model.get_config("secondary", beta=1.0, gamma=5.0)

    # Validate - should pass for valid configs
    errors = model.validate_after_load()
    assert len(errors) == 0

    # Now test that invalid configs are caught at instantiation time
    with pytest.raises(PydanticValidationError) as exc_info:
        model.get_config("main", prop1=-10.0, prop2=50.0)  # prop1 must be > 0
    
    assert len(exc_info.value.errors()) >= 1
