Plugin System
=============

Modern scientific and engineering workflows require flexible simulation frameworks that can be easily extended and customized.
NeoFOAM's plugin architecture is designed to enable users and developers to add new physics models, boundary conditions, and solver modules without modifying the core codebase.
This approach promotes maintainability, collaboration, and rapid prototyping of new features.

Overview
--------

The plugin system enables flexible model configuration via configuration files.
By utilizing Pydantic, the model configuration is decoupled from the specific file format, allowing support for multiple formats such as OpenFOAM dictionaries, YAML, TOML, or JSON.
It is a runtime-extensible system built on Pydantic discriminated unions and a registry pattern.

.. code-block:: cpp

    // RASModel acts as the discriminator
    // This tag/type field in the config file
    // selects the correct turbulence model
    RASModel        kOmegaSST;

    turbulence      on;

    printCoeffs     on;


A discriminated union selects the appropriate configuration class based on a type/tag field (e.g., "RASModel": "kOmegaSST" vs "RASModel": "kEpsilon").
Each base class maintains its own registry of registered child implementations.
All plugins and models are registered in a central registry, enabling easy access and management for UIs, validation, or generative AI integration.
This allows users to retrieve all available plugins (such as turbulence models or boundary conditions), inspect their configuration options, and validate them.
Additionally, the plugin system supports the generation of JSON schemas for documentation and validation.


Discriminated Unions in Pydantic
--------------------------------

The discriminated union functionality in Pydantic is illustrated
The discriminated union work in pydantic similar are given in the following simple example.

.. code-block:: python

    from typing import Literal, Union
    from pydantic import BaseModel, Field, ValidationError

    class Cat(BaseModel):
        # discriminator='pet_type'
        pet_type: Literal['cat']
        meows: int

    class Dog(BaseModel):
        pet_type: Literal['dog']
        barks: float

    class Lizard(BaseModel):
        pet_type: Literal['reptile', 'lizard']
        scales: bool

    class Model(BaseModel):
        # pet is the discriminator_variable
        pet: Union[Cat, Dog, Lizard] = Field(..., discriminator='pet_type')
        n: int

    print(Model(pet={'pet_type': 'dog', 'barks': 3.14}, n=1))
    #> pet=Dog(pet_type='dog', barks=3.14) n=1

Discriminated unions allow instantiating the correct subclass based on the value of a discriminator field (here ``pet_type``).
The ``discriminator_variable`` (here ``pet``) is the field that accepts the union of possible types.
Therefore, all available models must be registered in the Union to be selectable via the discriminator.
This facilitates easy validation of configuration files, as all required information is stored directly in the Pydantic models.

Plugin System and Registration of Subclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``PluginSystem`` stores and registers all base classes (similar to the ``Model`` shown above) in a centralized registry.
This ensures all plugins are easily accessible and usable by other parts of the codebase or external tools.

The main challenge lies in dynamically creating the Union type.
This is required because the plugin system enables the registration of new subclasses at runtime.

The example below demonstrates how to register a new base class with two subclasses using the ``PluginSystem``.
To automatically create the discriminated union, the base class must be decorated with ``@PluginSystem.register()``, providing the ``discriminator_variable`` and ``discriminator``.
This automatically adds the discriminated union field to the base class, similar to the Pydantic example above.

.. code-block:: python

    from pydantic import BaseModel
    from typing import Literal
    from foamadapter.core.plugin_system import PluginSystem

    @PluginSystem.register(discriminator_variable="shape", discriminator="shape_type")
    class ShapeInterface(BaseModel):
        # the decorator automatically adds shape as discriminator variable
        # and shape_type as discriminator
        # the commented line below is automatically added:
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


The ``ShapeInterface`` class now contains a dynamically created field ``shape``, which is a discriminated union of all registered subclasses.
The subclasses ``CircleConfig`` and ``SquareConfig`` are registered using the ``@ShapeInterface.register`` decorator.
This action automatically updates the union type and recreates the model to include the new subclasses.

The model must be instantiated using the class method ``create()`` to ensure that the latest version of the union type is used, as Pydantic models cache their schema definitions.

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


JSON Schema Generation
~~~~~~~~~~~~~~~~~~~~~~

The dynamically generated plugin models support JSON schema generation for validation and documentation.
JSON schemas expose all possible configurations for the available plugins of each base class, enabling the creation of user interfaces or configuration file validators.

``ShapeInterface.plugin_model`` provides access to the dynamically updated model and is required to generate up-to-date schemas.

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
