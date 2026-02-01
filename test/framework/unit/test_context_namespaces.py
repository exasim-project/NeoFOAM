# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Unit tests for Context namespace isolation.

Tests field vs model namespace separation and error handling.
"""

import pytest
from pydantic import ValidationError

from foamadapter.framework.context import Context, FieldUpdates


# ============================================================================
# Test: Context Creation
# ============================================================================


def test_context_creation_empty():
    """Test creating empty context."""
    ctx = Context(fields={}, models={})
    assert ctx.fields == {}
    assert ctx.models == {}
    assert ctx.mesh is None
    assert ctx.runTime is None


def test_context_creation_with_fields():
    """Test creating context with initial fields."""
    ctx = Context(fields={"temperature": 300.0, "pressure": 101325.0}, models={})
    assert "temperature" in ctx.fields
    assert "pressure" in ctx.fields
    assert ctx.fields["temperature"] == 300.0
    assert ctx.fields["pressure"] == 101325.0


def test_context_creation_with_models():
    """Test creating context with initial models."""

    class DummyModel:
        name = "test"

    model = DummyModel()
    ctx = Context(fields={}, models={"algorithm": model})
    assert "algorithm" in ctx.models
    assert ctx.models["algorithm"].name == "test"


def test_context_creation_with_mesh():
    """Test creating context with mesh."""

    class DummyMesh:
        nCells = 1000

    mesh = DummyMesh()
    ctx = Context(fields={}, models={}, mesh=mesh)
    assert ctx.mesh is not None
    assert ctx.mesh.nCells == 1000


# ============================================================================
# Test: Field Namespace
# ============================================================================


def test_field_access():
    """Test accessing fields from context."""
    ctx = Context(fields={"velocity": 10.0, "temperature": 300.0}, models={})
    assert ctx.fields["velocity"] == 10.0
    assert ctx.fields["temperature"] == 300.0


def test_field_update():
    """Test updating field values."""
    ctx = Context(fields={"velocity": 10.0}, models={})
    ctx.fields["velocity"] = 20.0
    assert ctx.fields["velocity"] == 20.0


def test_field_add():
    """Test adding new fields."""
    ctx = Context(fields={"velocity": 10.0}, models={})
    ctx.fields["pressure"] = 101325.0
    assert "pressure" in ctx.fields
    assert ctx.fields["pressure"] == 101325.0


def test_field_delete():
    """Test deleting fields."""
    ctx = Context(fields={"velocity": 10.0, "temperature": 300.0}, models={})
    del ctx.fields["temperature"]
    assert "temperature" not in ctx.fields
    assert "velocity" in ctx.fields


def test_field_iteration():
    """Test iterating over fields."""
    ctx = Context(
        fields={"velocity": 10.0, "temperature": 300.0, "pressure": 101325.0},
        models={},
    )
    field_names = list(ctx.fields.keys())
    assert len(field_names) == 3
    assert "velocity" in field_names
    assert "temperature" in field_names
    assert "pressure" in field_names


# ============================================================================
# Test: Model Namespace
# ============================================================================


def test_model_access():
    """Test accessing models from context."""

    class Algorithm:
        name = "PIMPLE"

    class Turbulence:
        model = "kEpsilon"

    ctx = Context(
        fields={},
        models={"algorithm": Algorithm(), "turbulence": Turbulence()},
    )
    assert ctx.models["algorithm"].name == "PIMPLE"
    assert ctx.models["turbulence"].model == "kEpsilon"


def test_model_update():
    """Test updating model references."""

    class OldAlgo:
        version = 1

    class NewAlgo:
        version = 2

    ctx = Context(fields={}, models={"algorithm": OldAlgo()})
    assert ctx.models["algorithm"].version == 1

    ctx.models["algorithm"] = NewAlgo()
    assert ctx.models["algorithm"].version == 2


def test_model_add():
    """Test adding new models."""

    class NewModel:
        name = "turbulence"

    ctx = Context(fields={}, models={})
    ctx.models["turbulence"] = NewModel()
    assert "turbulence" in ctx.models
    assert ctx.models["turbulence"].name == "turbulence"


def test_model_delete():
    """Test deleting models."""

    class Algo:
        pass

    class Turb:
        pass

    ctx = Context(fields={}, models={"algorithm": Algo(), "turbulence": Turb()})
    del ctx.models["turbulence"]
    assert "turbulence" not in ctx.models
    assert "algorithm" in ctx.models


# ============================================================================
# Test: Namespace Isolation
# ============================================================================


def test_field_model_namespace_isolation():
    """Test that fields and models are in separate namespaces."""

    class Algorithm:
        name = "solver"

    ctx = Context(fields={"name": "field_value"}, models={"name": Algorithm()})

    # Both can coexist with same key
    assert ctx.fields["name"] == "field_value"
    assert ctx.models["name"].name == "solver"


def test_field_namespace_does_not_leak_to_models():
    """Test fields namespace doesn't affect models."""
    ctx = Context(fields={"algorithm": "field_data"}, models={})

    assert "algorithm" in ctx.fields
    assert "algorithm" not in ctx.models


def test_model_namespace_does_not_leak_to_fields():
    """Test models namespace doesn't affect fields."""

    class Algo:
        pass

    ctx = Context(fields={}, models={"velocity": Algo()})

    assert "velocity" in ctx.models
    assert "velocity" not in ctx.fields


def test_nested_dict_access():
    """Test nested dictionaries in models."""
    ctx = Context(
        fields={},
        models={"config": {"param1": 1e-5, "param2": 1000.0, "nested": {"value": 42}}},
    )

    assert ctx.models["config"]["param1"] == 1e-5
    assert ctx.models["config"]["nested"]["value"] == 42


# ============================================================================
# Test: Error Handling
# ============================================================================


def test_access_missing_field_raises_keyerror():
    """Test accessing non-existent field raises KeyError."""
    ctx = Context(fields={}, models={})

    with pytest.raises(KeyError):
        _ = ctx.fields["nonexistent"]


def test_access_missing_model_raises_keyerror():
    """Test accessing non-existent model raises KeyError."""
    ctx = Context(fields={}, models={})

    with pytest.raises(KeyError):
        _ = ctx.models["nonexistent"]


def test_field_in_check():
    """Test 'in' operator for fields."""
    ctx = Context(fields={"velocity": 10.0}, models={})

    assert "velocity" in ctx.fields
    assert "pressure" not in ctx.fields


def test_model_in_check():
    """Test 'in' operator for models."""

    class Algo:
        pass

    ctx = Context(fields={}, models={"algorithm": Algo()})

    assert "algorithm" in ctx.models
    assert "turbulence" not in ctx.models


def test_get_with_default_fields():
    """Test get() with default value for fields."""
    ctx = Context(fields={"velocity": 10.0}, models={})

    assert ctx.fields.get("velocity") == 10.0
    assert ctx.fields.get("pressure", 101325.0) == 101325.0
    assert ctx.fields.get("missing") is None


def test_get_with_default_models():
    """Test get() with default value for models."""

    class Algo:
        name = "default"

    default_algo = Algo()
    ctx = Context(fields={}, models={"algorithm": Algo()})

    assert ctx.models.get("algorithm") is not None
    assert ctx.models.get("turbulence", default_algo).name == "default"
    assert ctx.models.get("missing") is None


# ============================================================================
# Test: FieldUpdates Integration
# ============================================================================


def test_field_updates_dict_behavior():
    """Test FieldUpdates behaves like a dict."""
    updates = FieldUpdates({"velocity": 20.0, "pressure": 200000.0})

    assert isinstance(updates, dict)
    assert updates["velocity"] == 20.0
    assert updates["pressure"] == 200000.0


def test_field_updates_creation():
    """Test creating FieldUpdates."""
    updates = FieldUpdates({"temp": 300.0})
    assert "temp" in updates
    assert updates["temp"] == 300.0


def test_apply_field_updates():
    """Test applying FieldUpdates to context."""
    ctx = Context(fields={"velocity": 10.0, "temperature": 300.0}, models={})

    updates = FieldUpdates({"velocity": 20.0, "pressure": 101325.0})
    ctx.fields.update(updates)

    assert ctx.fields["velocity"] == 20.0  # Updated
    assert ctx.fields["temperature"] == 300.0  # Unchanged
    assert ctx.fields["pressure"] == 101325.0  # Added


def test_field_updates_preserves_models():
    """Test FieldUpdates doesn't affect models."""

    class Algo:
        name = "algorithm"

    ctx = Context(fields={"velocity": 10.0}, models={"algorithm": Algo()})

    updates = FieldUpdates({"velocity": 20.0})
    ctx.fields.update(updates)

    assert ctx.fields["velocity"] == 20.0
    assert "algorithm" in ctx.models
    assert ctx.models["algorithm"].name == "algorithm"


# ============================================================================
# Test: Context Immutability Patterns
# ============================================================================


def test_context_field_copy():
    """Test copying field values."""
    ctx = Context(fields={"velocity": 10.0}, models={})

    # Get copy of value
    vel = ctx.fields["velocity"]
    vel = vel * 2

    # Original unchanged
    assert ctx.fields["velocity"] == 10.0


def test_context_model_reference():
    """Test model references (not copies)."""

    class Algo:
        value = 10

    algo = Algo()
    ctx = Context(fields={}, models={"algorithm": algo})

    # Get reference and modify
    algo_ref = ctx.models["algorithm"]
    algo_ref.value = 20

    # Original is modified (reference semantics)
    assert ctx.models["algorithm"].value == 20
    assert algo.value == 20


# ============================================================================
# Test: Complex Data Types
# ============================================================================


def test_field_with_list():
    """Test field containing list."""
    ctx = Context(fields={"values": [1.0, 2.0, 3.0]}, models={})
    assert ctx.fields["values"] == [1.0, 2.0, 3.0]
    assert len(ctx.fields["values"]) == 3


def test_field_with_dict():
    """Test field containing dictionary."""
    ctx = Context(fields={"config": {"tolerance": 1e-6, "maxIter": 100}}, models={})
    assert ctx.fields["config"]["tolerance"] == 1e-6
    assert ctx.fields["config"]["maxIter"] == 100


def test_model_with_nested_structure():
    """Test model with complex nested structure."""

    class ComplexModel:
        def __init__(self):
            self.params = {"a": 1, "b": 2}
            self.values = [10, 20, 30]

    model = ComplexModel()
    ctx = Context(fields={}, models={"complex": model})

    assert ctx.models["complex"].params["a"] == 1
    assert ctx.models["complex"].values[0] == 10


# ============================================================================
# Test: Context Attributes
# ============================================================================


def test_mesh_attribute():
    """Test mesh attribute separate from namespaces."""

    class Mesh:
        nCells = 1000

    ctx = Context(fields={}, models={}, mesh=Mesh())

    # Mesh is separate attribute
    assert ctx.mesh is not None
    assert ctx.mesh.nCells == 1000

    # Not in fields or models
    assert "mesh" not in ctx.fields
    assert "mesh" not in ctx.models


def test_runtime_attribute():
    """Test runTime attribute separate from namespaces."""

    class RunTime:
        time = 0.1

    ctx = Context(fields={}, models={}, runTime=RunTime())

    # RunTime is separate attribute
    assert ctx.runTime is not None
    assert ctx.runTime.time == 0.1

    # Not in fields or models
    assert "runTime" not in ctx.fields
    assert "runTime" not in ctx.models


def test_mesh_and_runtime_coexist():
    """Test mesh and runTime can coexist with field/model namespaces."""

    class Mesh:
        name = "mesh"

    class RunTime:
        name = "time"

    ctx = Context(
        fields={"mesh": "field_mesh", "runTime": "field_time"},
        models={"mesh": Mesh(), "runTime": RunTime()},
        mesh=Mesh(),
        runTime=RunTime(),
    )

    # All exist independently
    assert ctx.fields["mesh"] == "field_mesh"
    assert ctx.models["mesh"].name == "mesh"
    assert ctx.mesh.name == "mesh"

    assert ctx.fields["runTime"] == "field_time"
    assert ctx.models["runTime"].name == "time"
    assert ctx.runTime.name == "time"
