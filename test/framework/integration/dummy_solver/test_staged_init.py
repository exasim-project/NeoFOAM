# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for 3-stage initialization pattern.

Tests the StagedInit class and dummy_init_staged.py implementation
following the IncompressibleFluidInitializer pattern.
"""

import pytest


from foamadapter.framework.initialization import (
    StagedInit,
    ValidationError,
    ConfigContext,
)
from .dummy_init import create_init

# ============================================================================
# Test: StagedInit class basics
# ============================================================================


def test_staged_init_creation():
    """Test creating a StagedInit instance."""
    init = StagedInit("TestSolver")

    assert init.name == "TestSolver"
    assert init.argv == []
    assert init._load_func is None
    assert init._resolve_func is None
    assert init._build_func is None


def test_staged_init_decorators():
    """Test that decorators register functions."""
    init = StagedInit("TestSolver")

    @init.load
    def load_config():
        return {"test": "value"}

    @init.resolve
    def resolve_deps(config):
        pass

    @init.build
    def build_lazy():
        return []

    assert init._load_func is load_config
    assert init._resolve_func is resolve_deps
    assert init._build_func is build_lazy


def test_staged_init_requires_load():
    """Test that run() requires @init.load."""
    init = StagedInit("TestSolver")

    with pytest.raises(RuntimeError, match="No @TestSolver.load defined"):
        init.run()


def test_staged_init_requires_build():
    """Test that run() requires @init.build."""
    init = StagedInit("TestSolver")

    @init.load
    def load_config():
        return {}

    with pytest.raises(RuntimeError, match="No @TestSolver.build defined"):
        init.run()


# ============================================================================
# Test: Dummy solver 3-stage initialization
# ============================================================================


def test_dummy_init_staged_load():
    """Test LOAD stage of dummy_init_staged."""

    init_instance = create_init()
    load_result = init_instance.run_load()

    # Verify LoadResult structure
    assert hasattr(load_result, "core_models")
    assert hasattr(load_result, "optional_models")

    # Verify core models
    assert len(load_result.core_models) == 2
    algorithm, core_model2 = load_result.core_models

    # Verify algorithm
    assert algorithm is not None
    assert hasattr(algorithm, "solve")

    # Verify core_model2
    assert core_model2.name == "CoreModel2"
    assert core_model2.status == "active"

    # Verify optional models detected
    assert len(load_result.optional_models) > 0


def test_dummy_init_staged_validate_load():
    """Test LOAD stage validation."""

    init_instance = create_init()

    # Run load first
    load_result = init_instance.run_load()

    # Run validation
    errors = init_instance._validate_load_func(load_result.core_models)

    # Should have no errors with valid configuration
    error_count = len([e for e in errors if e.severity == "error"])
    assert error_count == 0


def test_dummy_init_staged_resolve():
    """Test RESOLVE stage of dummy_init_staged."""

    init_instance = create_init()

    # Run load first
    load_result = init_instance.run_load()

    # Build ConfigContext - register core models
    config = ConfigContext()
    for i, model in enumerate(load_result.core_models):
        config.register(f"core_model_{i}", model)

    # Run resolve
    init_instance.run_resolve(config)

    # Verify optional models can access config
    for model in load_result.optional_models:
        # Models should be able to access registered items
        assert config.get("core_model_0") is not None


def test_dummy_init_staged_validate_resolve():
    """Test RESOLVE stage validation."""

    init_instance = create_init()

    # Run load and resolve
    load_result = init_instance.run_load()
    config = ConfigContext()
    for i, model in enumerate(load_result.core_models):
        config.register(f"core_model_{i}", model)

    init_instance.run_resolve(config)

    # Run validation
    warnings = init_instance._validate_resolve_func(load_result.optional_models, config)

    # Should have no errors
    error_count = len([w for w in warnings if w.severity == "error"])
    assert error_count == 0


def test_dummy_init_staged_build():
    """Test BUILD stage of dummy_init_staged."""

    init_instance = create_init()

    # Run load and resolve first
    load_result = init_instance.run_load()
    config = ConfigContext()
    for i, model in enumerate(load_result.core_models):
        config.register(f"core_model_{i}", model)
    init_instance.run_resolve(config)

    # Run build
    lazy_inits = init_instance.run_build()

    # Verify lazy initializers were created
    assert len(lazy_inits) > 0

    # Check for expected initializers
    init_names = [li.name for li in lazy_inits]
    assert "mesh" in init_names
    assert "domain" in init_names
    assert "fields.field1" in init_names
    assert "fields.field2" in init_names
    assert "fields.field3" in init_names


def test_dummy_init_staged_full_run():
    """Test complete 3-stage initialization flow."""

    init_instance = create_init()
    init_instance.argv = []

    # Run full initialization
    ctx = init_instance.run()

    # Verify context has fields
    assert "field1" in ctx.fields
    assert "field2" in ctx.fields
    assert "field3" in ctx.fields

    # Verify field values
    assert ctx.fields["field1"] == 1.0
    assert ctx.fields["field2"] == 101325.0
    assert ctx.fields["field3"] == 0.01  # field1 * 0.01

    # Verify models
    assert "algorithm" in ctx.models
    assert "core2" in ctx.models
    assert "config" in ctx.models

    # Verify mesh
    assert "mesh" in ctx.mesh or hasattr(ctx, "mesh")


def test_dummy_init_staged_with_optional_models():
    """Test that optional models are initialized."""

    init_instance = create_init()
    init_instance.argv = []

    ctx = init_instance.run()

    # Verify model1 fields were created (if model1 is enabled)
    # These would be created by the optional model's build() method
    # Model fields are now in ctx.fields after the migration
    optional_model_keys = ["model_field1", "model_field2", "model_field3"]

    detected_optional_fields = [k for k in optional_model_keys if k in ctx.fields]

    # Should have at least some optional model fields if models were detected
    optional_models = getattr(init_instance, "_optional_models", [])
    if len(optional_models) > 0:
        assert len(detected_optional_fields) > 0


def test_dummy_init_staged_algorithm_configuration():
    """Test that optional models are properly initialized with new config API."""

    init_instance = create_init()
    init_instance.argv = []

    ctx = init_instance.run()

    algorithm = ctx.models.get("algorithm")
    assert algorithm is not None

    # If model1 is present, check that it has registered configs
    optional_models = getattr(init_instance, "_optional_models", [])
    model1 = next(
        (m for m in optional_models if hasattr(m, "name") and "Model1" in str(m.name)),
        None,
    )

    if model1:
        # Model1 should have configs registered via with_config()
        assert hasattr(model1, "_config_classes")
        assert "main" in model1._config_classes
        assert "step_config" in model1._config_classes


# ============================================================================
# Test: Validation error handling
# ============================================================================


def test_staged_init_load_error_handling():
    """Test that LOAD errors are raised."""
    init = StagedInit("TestSolver")

    @init.load
    def load_with_error():
        return {"algorithm": None}

    @init.validate_load
    def validate_with_error():
        return [ValidationError("algorithm", "Algorithm is None", severity="error")]

    @init.build
    def build_empty():
        return []

    with pytest.raises(RuntimeError, match="Load error: algorithm: Algorithm is None"):
        init.run()


def test_staged_init_load_warning_handling():
    """Test that LOAD warnings are printed but don't raise."""
    init = StagedInit("TestSolver")

    @init.load
    def load_with_warning():
        return {"algorithm": "dummy"}

    @init.validate_load
    def validate_with_warning():
        return [
            ValidationError(
                "optional_field", "Optional field missing", severity="warning"
            )
        ]

    @init.build
    def build_empty():
        return []

    # Should not raise - warnings are just printed
    ctx = init.run()
    assert ctx is not None


# ============================================================================
# Test: Factory function for dependency injection
# ============================================================================


def test_create_init_factory():
    """Test the create_init factory function."""
    from integration.dummy_solver.dummy_init import create_init

    init_instance = create_init()

    # Verify it's a StagedInit instance
    assert isinstance(init_instance, StagedInit)
    assert init_instance.name == "DummySolver"

    # Verify functions are registered
    assert init_instance._load_func is not None
    assert init_instance._resolve_func is not None
    assert init_instance._build_func is not None


def test_create_init_multiple_instances():
    """Test that create_init returns the same global instance."""
    from integration.dummy_solver.dummy_init import create_init

    init1 = create_init()
    init2 = create_init()

    # Should be the same instance (simplified factory)
    assert init1 is init2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
