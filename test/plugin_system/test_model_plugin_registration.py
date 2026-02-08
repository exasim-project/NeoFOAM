# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Test for registering Model() factory instances with PluginSystem.

Demonstrates how ModelInstance objects created with Model() factory can be
registered with PluginSystem using a chainable .register_with() method.

Pattern:
    model1 = Model("Model1").register_with(ModelInterface)

The model name is automatically used as the discriminator value for type-safe discovery.
"""

from pydantic import BaseModel
from typing import Any
from foamadapter.core.plugin_system import PluginSystem
from foamadapter.framework.model_factory import Model


@PluginSystem.register(discriminator_variable="config", discriminator="model_type")
class ModelInterface(BaseModel):
    """Base interface for registering models with PluginSystem."""

    enabled: bool = True
    name: str

    @classmethod
    def create(cls, *, config: dict[str, Any], **kwargs: Any) -> Any:
        """Factory classmethod for creating model instances."""
        return cls.plugin_model(config=config, **kwargs)  # type: ignore[attr-defined]


# Define and register models using chainable API
model1 = Model("Model1").register_with(ModelInterface)


@model1.detect
def detect_model1() -> bool:
    """Always enable Model 1."""
    return True


model2 = Model("Model2").register_with(ModelInterface)


@model2.detect
def detect_model2() -> bool:
    """Always enable Model 2."""
    return True


# ============================================================================
# Tests
# ============================================================================


def test_model_factory_instances_registered() -> None:
    """Test that ModelInstance objects created with Model() factory are registered."""
    registry = PluginSystem.get_registered("ModelInterface")
    assert registry is not None
    plugin_names = [cls.__name__ for cls in registry.plugin_registry]
    assert "Model1" in plugin_names
    assert "Model2" in plugin_names
    assert len(plugin_names) == 2


def test_model_factory_create_method() -> None:
    """Test that factory create method works correctly."""
    m1 = ModelInterface.create(
        config={"model_type": "Model1"},
        enabled=True,
        name="Created1",
    )
    assert isinstance(m1, BaseModel)
    assert m1.config.model_type == "Model1"
    assert m1.name == "Created1"
    assert m1.enabled is True

    m2 = ModelInterface.create(
        config={"model_type": "Model2"},
        enabled=False,
        name="Created2",
    )
    assert isinstance(m2, BaseModel)
    assert m2.config.model_type == "Model2"
    assert m2.enabled is False


def test_list_model_plugins() -> None:
    """Test listing all registered model plugins."""
    plugins = PluginSystem.list_plugins()
    assert "ModelInterface" in plugins
    model_plugin_names = [cls.__name__ for cls in plugins["ModelInterface"]]
    assert "Model1" in model_plugin_names
    assert "Model2" in model_plugin_names
    assert len(model_plugin_names) == 2


def test_json_schema_generation() -> None:
    """Test that JSON schema is generated correctly with discriminator."""
    ModelPlugin = ModelInterface.plugin_model  # type: ignore[attr-defined]
    schema = ModelPlugin.model_json_schema()
    discriminator = schema["properties"]["config"]["discriminator"]
    assert discriminator["propertyName"] == "model_type"
    mapping = discriminator["mapping"]
    assert "Model1" in mapping
    assert "Model2" in mapping
    assert len(mapping) == 2
