"""
Test for an extensible plugin config system using pydantic discriminated unions and a registry pattern.
Refactored to use a generic registry and factory for multiple extensible models.
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Literal
import pytest
from foamadapter.core.plugin_system import PluginSystem


@PluginSystem.register(discriminator_variable="shape", discriminator="shape_type")
class ShapeInterface(BaseModel):
    color: str


@ShapeInterface.register
class CircleConfig(BaseModel):
    shape_type: Literal["circle"]
    radius: float


@ShapeInterface.register
class SquareConfig(BaseModel):
    shape_type: Literal["square"]
    side: float


@ShapeInterface.register
class RectangleConfig(BaseModel):
    shape_type: Literal["rectangle"]
    width: float
    height: float


# Extensible types (added in tests)
class TriangleConfig(BaseModel):
    shape_type: Literal["triangle"]
    base: float
    height: float


class PolygonConfig(BaseModel):
    shape_type: Literal["polygon"]
    sides: int
    length: float


Shape = ShapeInterface.plugin_model


def test_valid_configs_n_models():
    m1 = Shape(shape={"shape_type": "circle", "radius": 2.5}, color="red")
    assert m1.shape.radius == 2.5
    m2 = Shape(shape={"shape_type": "square", "side": 4.0}, color="blue")
    assert m2.shape.side == 4.0
    m3 = Shape(
        shape={"shape_type": "rectangle", "width": 3.0, "height": 6.0}, color="green"
    )
    assert m3.shape.width == 3.0
    assert m3.shape.height == 6.0


def test_invalid_config_n_models():
    with pytest.raises(ValidationError):
        Shape(shape={"shape_type": "circle"}, color="bad")  # missing 'radius'
    with pytest.raises(ValidationError):
        Shape(shape={"shape_type": "square", "radius": 2.0}, color="bad")  # wrong field


def test_extensibility_n_models():
    ShapeInterface.register(TriangleConfig)
    m4 = ShapeInterface.create(
        shape={"shape_type": "triangle", "base": 3.0, "height": 4.0}, color="yellow"
    )
    assert m4.shape.base == 3.0
    assert m4.shape.height == 4.0
    assert m4.shape.shape_type == "triangle"

    ShapeInterface.register(PolygonConfig)
    m5 = ShapeInterface.create(
        shape={"shape_type": "polygon", "sides": 5, "length": 2.0}, color="purple"
    )
    assert m5.shape.sides == 5
    assert m5.shape.length == 2.0
    assert m5.shape.shape_type == "polygon"

    assert PluginSystem.remove_plugin_model("ShapeInterface", TriangleConfig) is True
    assert PluginSystem.remove_plugin_model("ShapeInterface", PolygonConfig) is True


def test_json_schema():
    Shape = ShapeInterface.plugin_model
    schema = Shape.model_json_schema()
    discriminator = schema["properties"]["shape"]["discriminator"]
    assert discriminator["propertyName"] == "shape_type"
    mapping = discriminator["mapping"]
    assert "circle" in mapping
    assert "square" in mapping
    assert "rectangle" in mapping
    assert "triangle" not in mapping
    assert "polygon" not in mapping

    ShapeInterface.register(TriangleConfig)
    Shape = ShapeInterface.plugin_model
    schema = Shape.model_json_schema()
    mapping = schema["properties"]["shape"]["discriminator"]["mapping"]
    assert "triangle" in mapping
    assert "polygon" not in mapping

    ShapeInterface.register(PolygonConfig)
    Shape = ShapeInterface.plugin_model
    schema = Shape.model_json_schema()
    mapping = schema["properties"]["shape"]["discriminator"]["mapping"]
    assert "polygon" in mapping

    assert PluginSystem.remove_plugin_model("ShapeInterface", TriangleConfig) is True
    assert PluginSystem.remove_plugin_model("ShapeInterface", PolygonConfig) is True


def test_plugin_registry():
    ShapeInterface.register(TriangleConfig)
    ShapeInterface.register(PolygonConfig)
    registry = PluginSystem.get_registered("ShapeInterface")
    assert registry is not None
    plugin_names = [cls.__name__ for cls in registry.plugin_registry]
    assert "CircleConfig" in plugin_names
    assert "SquareConfig" in plugin_names
    assert "RectangleConfig" in plugin_names
    assert "TriangleConfig" in plugin_names
    assert "PolygonConfig" in plugin_names

    assert PluginSystem.remove_plugin_model("ShapeInterface", TriangleConfig) is True
    assert PluginSystem.remove_plugin_model("ShapeInterface", PolygonConfig) is True


@PluginSystem.register(discriminator_variable="animal", discriminator="animal_type")
class AnimalInterface(BaseModel):
    color: str

@AnimalInterface.register
class Dog(BaseModel):
    animal_type: Literal["dog"]


@AnimalInterface.register
class Cat(BaseModel):
    animal_type: Literal["cat"]

def test_list_plugins():
    ShapeInterface.register(TriangleConfig)
    ShapeInterface.register(PolygonConfig)
    plugins = PluginSystem.list_plugins()
    assert "ShapeInterface" in plugins
    shape_plugin_names = [cls.__name__ for cls in plugins["ShapeInterface"]]
    assert "CircleConfig" in shape_plugin_names
    assert "SquareConfig" in shape_plugin_names
    assert "RectangleConfig" in shape_plugin_names
    assert "TriangleConfig" in shape_plugin_names
    assert "PolygonConfig" in shape_plugin_names

    assert PluginSystem.remove_plugin_model("ShapeInterface", TriangleConfig) is True
    assert PluginSystem.remove_plugin_model("ShapeInterface", PolygonConfig) is True

    animal_plugin_names = [cls.__name__ for cls in plugins["AnimalInterface"]]
    assert "Dog" in animal_plugin_names
    assert "Cat" in animal_plugin_names
