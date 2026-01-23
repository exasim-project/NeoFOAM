Plugin System
=============

Modern scientific and engineering workflows require flexible simulation frameworks that can be easily extended and customized.
NeoFOAM's plugin architecture is designed to enable users and developers to add new physics models, boundary conditions, and solver modules without modifying the core codebase.
This approach promotes maintainability, collaboration, and rapid prototyping of new features.

Overview
--------

The PluginSystem is a runtime-extensible configuration system built on Pydantic discriminated unions and a registry pattern.
A discriminated union selects the right config class based on a type/tag field (e.g., "shape_type": "circle" vs "shape_type": "square"), and each base class has its own registry of registered child implementations.
All plugins and all models are registered in a central registry to enable easy access and management for UI,validation purposes or generative AI.
The user would be able to retrieve all available plugins: turbulence models, boundary conditions, etc. and their configuration options and validate them.
The system also supports the generation of JSON schemas for documentation and validation purposes.

Registration Process
~~~~~~~~~~~~~~~~~~~~

The decorator-based registration follows a two-step process:

1. **Base Class Registration**: ``@PluginSystem.register()`` creates a registry entry and adds helper methods
2. **Plugin Registration**: ``@BaseClass.register`` adds plugin classes to the registry and regenerates the union model

Dynamic Model Generation
~~~~~~~~~~~~~~~~~~~~~~~~

The system uses ``pydantic.create_model()`` to dynamically generate extensible models.
Each registration updates the discriminated union type and recreates the model.
The discriminator field enables automatic deserialization based on the type identifier.

Usage Examples
--------------

Basic Plugin Setup
~~~~~~~~~~~~~~~~~~

The following example demonstrates how to define a plugin base class and register multiple plugin configurations.

.. code-block:: python

    from pydantic import BaseModel
    from typing import Literal
    from foamadapter.core.plugin_system import PluginSystem

    @PluginSystem.register(discriminator_variable="shape", discriminator="shape_type")
    class ShapeInterface(BaseModel):
        # the decorator automatically adds shape as discriminator field
        # shape: Union[CircleConfig, SquareConfig] = Field(discriminator='shape_type')
        color: str
        name: str = "default"

    @ShapeInterface.register
    class CircleConfig(BaseModel):
        shape_type: Literal["circle"]
        radius: float

    @ShapeInterface.register
    class SquareConfig(BaseModel):
        shape_type: Literal["square"]
        side: float

Creating and Using Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A usage example creating and using plugin configurations is shown below.

.. code-block:: python

    # return a circle instance
    circle = ShapeInterface.create(
        shape={"shape_type": "circle", "radius": 5.0}, # shows the ability to serialize via discriminator
        color="red"
    )

    # return a square instance
    square = ShapeInterface.create(
        shape={"shape_type": "square", "side": 3.0},
        color="blue"
    )

    # Accessing plugin-specific fields
    assert circle.shape.radius == 5.0
    assert square.shape.side == 3.0

Runtime Extension
~~~~~~~~~~~~~~~~~

The plugin system supports runtime registration of new plugin configurations.
However, this requires rebuilding the Type union as the models number of registered types changes.
Otherwise, the newly added types would not be part of the union and could not be registered and selected.

.. code-block:: python

    # Define new plugin configurations
    class TriangleConfig(BaseModel):
        shape_type: Literal["triangle"]
        base: float
        height: float

    class EllipseConfig(BaseModel):
        shape_type: Literal["ellipse"]
        major_axis: float
        minor_axis: float

    # Register at runtime
    ShapeInterface.register(TriangleConfig) # register the new plugin
    ShapeInterface.register(EllipseConfig) # register the new plugin

    # Use new configurations immediately and internally rebuild the union model
    triangle = ShapeInterface.create(
        shape={"shape_type": "triangle", "base": 4.0, "height": 3.0},
        color="green"
    )

Registering an additional base class can be simply done by decorating it with ``@PluginSystem.register()``.
Internally, PluginSystem would store the registries in similar to a dictionary

.. code-block:: python

    # pseudo-code representation of internal registries
    {
        "ShapeInterface": <Registry for ShapeInterface>,
        "OtherInterface": <Registry for OtherInterface>
    }



JSON Schema Generation
~~~~~~~~~~~~~~~~~~~~~~

The dynamically generated plugin models support JSON schema generation for validation and documentation purposes.
Json schemas show all possible configuration of all the available plugins of each base class and therefore enable the creation of user interfaces or config file validators.

.. code-block:: python

    # Get JSON schema for validation and documentation
    Shape = ShapeInterface.plugin_model
    schema = Shape.model_json_schema()

    # Schema includes discriminator mapping
    discriminator = schema["properties"]["shape"]["discriminator"]
    print(discriminator["propertyName"])  # "shape_type"
    print(discriminator["mapping"])       # {"circle": "...", "square": "..."}

Registry Management
~~~~~~~~~~~~~~~~~~~

The PluginSystem class provides utility methods to manage and inspect the plugin registries.
The following example demonstrates how to list registered plugins, retrieve specific registries, and remove plugins.

.. code-block:: python

    # List all plugins
    all_plugins = PluginSystem.list_plugins()
    print(all_plugins.keys())  # ["turbulence models", "thermodynamic models", ...]

    # Get specific registry
    shape_registry = PluginSystem.get_registered("TurbulenceModel")
    plugin_classes = shape_registry.plugin_registry

    # Remove plugins
    success = PluginSystem.remove_plugin_model("TurbulenceModel", KOmegaSSTModel)

Registration via Entrypoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. warning::

    This feature is not yet implemented.
