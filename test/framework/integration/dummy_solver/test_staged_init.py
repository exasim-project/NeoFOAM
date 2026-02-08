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


# ============================================================================
# Test: Dummy solver 3-stage initialization
# ============================================================================


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


def test_optional_models_integration():
    """Test that optional models are properly initialized and configured."""

    init_instance = create_init()
    init_instance.argv = []

    ctx = init_instance.run()

    # Verify model fields were created (if models are enabled)
    optional_model_keys = ["model_field1", "model_field2", "model_field3"]
    detected_optional_fields = [k for k in optional_model_keys if k in ctx.fields]

    # Should have optional model fields if models were detected
    optional_models = getattr(init_instance, "_optional_models", [])
    if len(optional_models) > 0:
        assert len(detected_optional_fields) > 0

    # Verify algorithm
    algorithm = ctx.models.get("algorithm")
    assert algorithm is not None

    # If model1 is present, check that it has registered configs
    model1 = next(
        (m for m in optional_models if hasattr(m, "name") and "Model1" in str(m.name)),
        None,
    )

    if model1:
        # Model1 should have configs loaded via auto-discovery
        assert hasattr(model1, "_configs")
        assert len(model1._configs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
