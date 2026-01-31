Plugin System
=============

Modern scientific and engineering workflows require flexible simulation frameworks that can be easily extended and customized.
NeoFOAM's plugin architecture is designed to enable users and developers to add new physics models, boundary conditions, and solver modules without modifying the core codebase.
This approach promotes maintainability, collaboration, and rapid prototyping of new features.

Overview
--------

The plugin system allows the configuration of the used models via configuration files and is a runtime-extensible configuration system built on Pydantic discriminated unions and a registry pattern.

.. code-block:: cpp

    // RASModel would be discriminator
    // a tags / type field in the config file
    // that selects the right turbulence model
    RASModel        kOmegaSST;

    turbulence      on;

    printCoeffs     on;


A discriminated union selects the right config class based on a type/tag field (e.g., "RASModel": "kOmegaSST" vs "RASModel": "kEpsilon"), and each base class has its own registry of registered child implementations.
All plugins and all models are registered in a central registry to enable easy access and management for UI, validation purposes or generative AI.
So, the user can retrieve all available plugins, like turbulence models, boundary conditions, etc.  their configuration options and is able to validate them.
Additionally, the plugin system supports the generation of JSON schemas for documentation and validation purposes.


Discrimated Unions in Pydantic
------------------------------

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

They allow to instantiate the correct subclass based on the value of a discriminator field (here ``pet_type``).
The ``discriminator_variable`` (here ``pet``) is the field that holds the union of possible types.
So, all available models must be registered in the Union to be selectable via the discriminator.
This is allows to easily validate the configuration files all required information is stored in the pydantic models.

Plugin System and Registration of subclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PluginSystem stores and register all Base classes similar to the ``Model`` shown above into a single class.
This way all plugins are easily accessible and usable by other parts of the codebase or external tools.

The main challenge is to dynamically create the Union type. This is required as the plugin is able to register new subclasses at runtime.

The example below shows how to register a new base class with two subclasses using the PluginSystem.
To be able to automatically create the discriminated union, the base class must be decorated with ``@PluginSystem.register()``.
and the ``discriminator_variable`` and ``discriminator`` must be provided.
This will automatically add the discriminated union field to the base class similar to the example above.

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


The ``ShapeInterface`` class now has a dynamically created field ``shape`` that is a discriminated union of all registered subclasses.
The subclasses ``CircleConfig`` and ``SquareConfig`` are registered using the ``@ShapeInterface.register`` decorator.
This automatically updates the union type and recreates the model.

The model need to be create by the class method ``create()`` to ensure that the latest version of the union type is used.

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

The dynamically generated plugin models support JSON schema generation for validation and documentation purposes.
Json schemas show all possible configuration of all the available plugins of each base class and therefore enable the creation of user interfaces or config file validators.

``ShapeInterface.plugin_model`` provides access to the dynamically update model and is required to generate up-to-date schemas.

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
