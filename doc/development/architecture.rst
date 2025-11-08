Architecture
============

FoamAdapter is multilanguage repository as it contains both C++ and python code.
This document describes the architecture of FoamAdapter, including the C++ core and Python interface components.

.. note::
   This section of the documentation should provide:
     * a high-level overview of the *planned* architecture 
     * guidance through the review process
     * The example implementation only serves as proof of concept and are only should provide a impression of the *planned* architecture.
     * The implementation of the detailed features will update the following sections and will further refine the architecture and change the code examples.

Overview
--------

FoamAdapter is designed as a multi-physics, python-based simulation framework.
Additionally, a set of C++ based legacy solvers like neoIcoFoam simplify the transition from existing OpenFOAM solvers and workflows.
It provides a flexible and modular architecture that allows users to easily extend and customize the simulation setup.

The architecture provides following features to achieve the goals outlined in the :doc:`goals and features <../goals_features>` document:

- Easy coupling of multiple domains and physics
- Field and model initialization based on dependency graphs
- Modular solver design that computes the data dependencies at runtime
- Plugin architecture for easy extension with new models and fields

In order to implement multi-physics capabilities multiple computational domains are supported.  
Each computational domain has **one solver** assigned that defines the governing equations, operations and **optional additional physical models.**
Multiple domains can be defined with input files and the coupling between the domains is handled automatically based on the defined physics modules.

The following sections describe the main feature and implementation example to give a high level overview of the architecture.

Extensible Solver Architecture
------------------------------

To foster code reuse and maintainability, the execution steps (operations) can be configured at runtime based on the selected physics models.
This is illustrated in the following diagram, where the fluid solver is extended by three physics submodules.

.. mermaid::

   flowchart TD
        
        subgraph MAIN ["Main Solver Loop"]
            STEP1["Solver </br> Momentum equation"]
            STEP2["add by model </br> Temperature equation"]
            STEP3["Solver </br> continuity equation"]
            STEP4["Solver </br> update turbulence"]
        end
        
        STEP1 --> STEP2
        STEP2 --> STEP3
        STEP3 --> STEP4
        
        %% Physics Extensions (simplified)
        subgraph AddPhysics ["Additional Physics Modules"]
            direction TB
            POROSITY["Porosity"]
            ROTATION["Rotating Reference Frame"]
            BUOYANCY["Boussinesq Approximation"]
        end
        POROSITY -.-> STEP1
        ROTATION -.-> STEP1
        BUOYANCY -.-> STEP2
        BUOYANCY -.-> STEP3

        style MAIN fill:#E3F2FD
        style AddPhysics fill:#E3F2FD
        style STEP1 fill:#2196F3,color:#fff
        style STEP2 fill:#FF9800,color:#fff
        style STEP3 fill:#9C27B0,color:#fff
        style STEP4 fill:#607D8B,color:#fff

The solver defines the main operations to be executed: Momentum, continuity and the turbulence model update.
Additional physics models such as porosity, rotation, buoyancy can modify/add the main operations.
This relation is defined defined in the pseudocode below:

.. code-block:: python
    # Pseudocode illustrating the solver and model structure
    @Solver
    class IncompressibleFluidSolver:
        models: list[IncompressibleFluidModel]  # List of additional physics models
        @Solver.step(...)
        def momentum(self, ...): pass
        @Solver.step(...)
        def continuity(self, ...): pass
        @Solver.step(...)
        def update_turbulence(self, ...): pass

    @IncompressibleFluidModel.register
    class BoussinesqModel:
        @IncompressibleFluidModel.step(...)
        def temperature_equation(self, ...): pass

Each solver has a list of additional physics models that can add/modify the solver operations (see diagram above).
Therefore, the solver can be easily extended with new physics models with little extra code.
The user just needs to define a new physics model that adds the desired operations to the solver at runtime and without modifying the core solver implementation.
The new models can be registered via the plugin architecture described later and selected via input files.

Conceptual Implementation
-------------------------

After the solver and models are initialized, operations need to be identified, and sorted to determine the correct execution order of the solver and models.
The correct execution order of the operations are stored in the **Operations** class roughly sketched below:

.. code-block:: python
    # Pseudocode how the operations are stored and executed
    class Operations:
        ops: list[Operation]  # All operations to be executed

        def run(self,...): pass

    class Operation:
        func: callable
        sub_operations: list[Operation]  # Operations added by models
        metadata: Any  # Additional metadata to describe/sort the operation

        def run(self,...): pass
 
An operation represents a single computational step in the solver or model and can be seen as a function that performs a specific task.
So, each solver or model can define multiple operations that are stored as Operation.
This class is able to hold sub operations and metadata to help with the sorting the process of the operations.

After the sorting process is completed (that will be described in detail in future documentation), the **Operations** class holds all the steps to sucessfully run the modified solver.

This modular design enables users to easily add or remove physics effects without altering the fundamental solver structure, promoting code reuse and maintainability.


.. note::

    implementation is still work in progress

Simulation with multiple Domains/Solvers
----------------------------------------

This concept can be easily extended to a multi-physics scenario with multiple computational domains and solvers.
Each solver would define its own operations and the coupling between the solver can be automatically managed at runtime based on the defined physics models/settings.

The following diagram illustrates the workflow for two common solver types for a conjugate heat transfer scenario:

.. mermaid::

    %%{init: {'flowchart': { 'htmlLabels': true, 'wrap': true }}}%%
    flowchart TB
        subgraph setup1 ["setup "]
            direction TB
            S1A["Initialize Fields"]
        end

        subgraph setup2 ["setup"]
            direction TB
            S2A["Initialize Temperature Field"]
        end

        subgraph S1 ["non-thermal-fluid solver"]
            direction TB
            S1C["Momentum Predictor"]
            S1C --> S1E["Solve Energy Equation"]
            S1E --> S1F["Pressure Corrector<br/>PISO Loop"]
            S1F --> S1J["Update turbulence Model"]
        end

        subgraph S2 ["thermal-solid solver"]
            direction TB
            S2B["Solve Energy Equation"] --> S2D["Update Solid Properties"]
        end

        setup1 --> S1C
        setup2 --> S2B
        S1E --> S2B
        S2B --> S1E


The diagram shows two selected solvers: a non-thermal-fluid solver and a thermal-solid solver that solve a conjugate heat transfer problem.
Each solver has its own initialization phase after this the **Operations** of each solver are determined based on the selected physics models.
One the coupling between the **Operations** is determined, the Operations of both solvers are modfied to include the data exchange between the two solvers.


.. note::

    The implementation is still work in progress.


Initialization Stage
--------------------


The initialization stage is responsible for setting up the simulation environment, including reading input files, initializing fields, and preparing solvers and models.

It is not fully designed yet but a stage initiliation is envisioned that performs the following tasks:

1. Read input files and parse configuration settings.
2. Initialize computational domains and meshes.
3. Exchange dependencies between the different operations e.g. adding a Boussinesq Approximation requires to solve a different formaulation in the pressure equation.
4. Define the operations for each solver based on the selected physics models.
5. Sort the operations to determine the correct execution order.


Plugin Architecture
-------------------

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

Pydantic supports discriminated unions for type-safe configuration, but the set of types in the union must be known at model definition time. 
The following example ensures that a pet is either Cat, Dog, or Lizard

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


Model Introspection and Schema Generation
------------------------------------------

Model configuration and validation in FoamAdapter are implemented using Pydantic, which provides native support for input validation and automatic JSON Schema generation.
This mechanism forms the basis for model discovery, UI integration, and automated documentation across the framework.

Pydantic’s schema generation enables the following functionality:

* Input validation: Ensures model configurations are consistent and type-safe.
* UI integration: Allows user interfaces to be generated dynamically from model definitions.
* Automatic documentation: Exposes field names, types, and constraints for all models.
* Metadata generation: Facilitates downstream tools to query and reason about model structures.
* AI-assisted workflows: Supports schema-driven interactions with generative AI systems.

To obtain a model’s JSON Schema representation, use the model_json_schema() method provided by Pydantic:

.. code-block:: python

    # For a registered plugin or configuration model
    schema = ShapeBase.plugin_model.model_json_schema()

    # For any Pydantic model
    schema = MyModel.model_json_schema()

This interface provides a uniform mechanism for introspection of all models in FoamAdapter, making it possible to programmatically discover available fields, their data types, validation rules, and default values.

This requires that all plugin, solver, or model use Pydantic to configure the inputs.


