
Plugin System
=============

Motivation
^^^^^^^^^^

Modern scientific and engineering workflows require flexible simulation frameworks that can be easily extended and customized.
FoamAdapter's plugin architecture is designed to enable users and developers to add new physics models, boundary conditions, and solver modules without modifying the core codebase.
This approach promotes maintainability, collaboration, and rapid prototyping of new features.



Concept
^^^^^^^
FoamAdapter implements a runtime-extensible plugin/config system using Pydantic discriminated unions and a registry pattern.
The core idea is to allow new plugin types (e.g., models, fields, solvers) to be registered dynamically, either at runtime or via Python entry points (setuptools).
Each plugin type (such as physics models or boundary conditions) is managed by a registry, which collects all available plugin classes and exposes a unified configuration model for input validation and schema generation.

**Background: Pydantic Discriminated Unions**

Pydantic supports discriminated unions for type-safe configuration, but the set of types in the union must be known at model definition time. For example:

.. code-block:: python

    from typing import Literal, Union
    from pydantic import BaseModel, Field

    class Cat(BaseModel):
        pet_type: Literal['cat']
        meows: int

    class Dog(BaseModel):
        pet_type: Literal['dog']
        barks: float

    class Lizard(BaseModel):
        pet_type: Literal['reptile', 'lizard']
        scales: bool

    class Model(BaseModel):
        pet: Union[Cat, Dog, Lizard] = Field(discriminator='pet_type')
        n: int

This works well for static unions, but it is not possible to add new types to the union at runtime. This is a challenge for plugin systems, where extensibility is required.

**How FoamAdapter Solves This**

Plugins are registered using a decorator-based API, making it easy for users to define and integrate new modules.
Whenever a new plugin is registered, the system automatically rebuilds the Pydantic model for the plugin type, updating the discriminated union to include all registered types.
This means that the configuration model always reflects the current set of available plugins, and input validation is always up to date.

For example, after registering a new shape plugin, you can immediately use the updated model for validation:

.. code-block:: python

    ShapeBase.register(TriangleConfig)
    shape = ShapeBase.plugin_model(shape={"shape_type": "triangle", "base": 3.0, "height": 4.0}, color="yellow")

This dynamic rebuilding of the model enables true runtime extensibility and ensures that input validation and schema generation always match the available plugins.
The `plugin_model` attribute needs to be called to obtain the up-to-date model for the plugin type.

Usage
^^^^^

To add a new plugin, users simply define a new Python class for their model or field and register it with the appropriate base class:

.. code-block:: python

    from foamadapter.core.plugin_system import PluginSystem

    @PluginSystem.register(discriminator_variable="model", discriminator="model_type")
    class ModelBase(BaseModel):
        name: str

    @ModelBase.register
    class MyCustomModel(BaseModel):
        model_type: Literal["custom"]
        parameter: float

    # Instantiate a model config
    config = ModelBase.create(model={"model_type": "custom", "parameter": 1.23}, name="example")

Plugins can also be discovered and registered automatically via Python entry points, allowing third-party packages to extend FoamAdapter seamlessly.
The unified configuration model and schema make it easy to build UIs, validate inputs, and document available plugins.




