Plugin System
=============

Modern scientific and engineering workflows require flexible simulation frameworks that can be easily extended and customized.
FoamAdapter's plugin architecture is designed to enable users and developers to add new physics models, boundary conditions, and solver modules without modifying the core codebase.
This approach promotes maintainability, collaboration, and rapid prototyping of new features.

Overview
--------

The PluginSystem provides a runtime-extensible configuration system using Pydantic discriminated unions and a registry pattern.
The system allows to register child classes on numerous base classes, each with its own registry.

Core Components
---------------

PluginRegistry Dataclass
~~~~~~~~~~~~~~~~~~~~~~~~

The ``PluginRegistry`` dataclass stores metadata for each plugin base type:

* ``base_cls``: The plugin base class (Pydantic BaseModel)
* ``discriminator_variable``: Field name holding the union (e.g., 'shape', 'plugin')
* ``discriminator``: Discriminator field name in plugin configs (e.g., 'shape_type', 'plugin_type')
* ``plugin_registry``: List of registered plugin configuration classes
* ``plugin_model``: Dynamically generated extensible Pydantic model

PluginSystem Class
~~~~~~~~~~~~~~~~~~

The central registry stores all plugin families within the class dictionary variable ``_registry`` .
Each key represents a plugin family name mapping to its corresponding PluginRegistry instance.

Implementation Details
----------------------

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

    # Direct instantiation using the plugin model
    Shape = ShapeInterface.plugin_model
    circle = Shape(
        shape={"shape_type": "circle", "radius": 5.0},
        color="red"
    )

    # Using the create class method
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
However, this requires rebuilding the Type union which is done by the create method.

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
    ShapeInterface.register(TriangleConfig)
    ShapeInterface.register(EllipseConfig)

    # Use new configurations immediately
    triangle = ShapeInterface.create(
        shape={"shape_type": "triangle", "base": 4.0, "height": 3.0},
        color="green"
    )

Register multiple classes
~~~~~~~~~~~~~~~~~~~~~~~~~

`AnimalInterface` will be stored in a central registry to show the availability of all registered plugins.

.. code-block:: python

    @PluginSystem.register(discriminator_variable="animal", discriminator="animal_type")
    class AnimalInterface(BaseModel):
        name: str
        age: int

    @AnimalInterface.register
    class DogConfig(BaseModel):
        animal_type: Literal["dog"]
        breed: str
        is_trained: bool = False

    @AnimalInterface.register
    class CatConfig(BaseModel):
        animal_type: Literal["cat"]
        indoor_only: bool = True
        declawed: bool = False

    # Independent from ShapeInterface family
    my_dog = AnimalInterface.create(
        animal={"animal_type": "dog", "breed": "Golden Retriever", "is_trained": True},
        name="Buddy",
        age=3
    )


    

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

    # List all plugin families
    all_plugins = PluginSystem.list_plugins()
    print(all_plugins.keys())  # ["ShapeInterface", "DataProcessor"]

    # Get specific registry
    shape_registry = PluginSystem.get_registered("ShapeInterface")
    plugin_classes = shape_registry.plugin_registry

    # Remove plugins
    success = PluginSystem.remove_plugin_model("ShapeInterface", TriangleConfig)

Registeration via Entrypoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. warning::

    This feature is not yet implemented.

