Architecture
============

FoamAdapter is a multi-language repository containing both C++ and Python code.
This document describes the architecture of FoamAdapter, including its C++ core and Python interface components.

.. note::
   This section of the documentation provides:
     * a high-level overview of the *planned* architecture
     * guidance through the review process
     * example implementations serving only as proof of concept to illustrate the *planned* architecture
     * a note that detailed features will evolve and refine the architecture, updating examples as development progresses

Overview
--------

NeoFOAM is designed as a multi-physics, Python-based simulation framework.
Additionally, it includes a set of C++-based legacy solvers, such as neoIcoFoam, to simplify the transition from existing OpenFOAM solvers and workflows.
It provides a flexible and modular architecture that allows users to easily extend and customize their simulation setup.

The architecture provides the following features to achieve the goals outlined in the :doc:`goals and features <../goals_features>` document:

- Easy coupling of multiple domains and physical models
- Field and model initialization based on dependency graphs
- Modular solver design that computes data dependencies at runtime
- Plugin architecture for extending models and fields

To support multi-physics capabilities, multiple computational domains are supported.
Each domain has **one solver** assigned, which defines the governing equations, operations, and **optional additional physical models.**
Coupling between domains is automatically handled based on the selected physics modules.

The following sections describe the main architectural features and implementation examples, providing a high-level overview.

Extensible Solver Architecture
------------------------------

To promote code reuse and maintainability, solver execution operations can be configured at runtime based on the selected physics models.
This is illustrated in the diagram below, where a fluid solver is extended with three physics submodules.

.. mermaid::

   flowchart TD

        subgraph MAIN ["Main Solver Loop"]
            OP1["Solver </br> Momentum Equation"]
            OP2["Added by Model </br> Temperature Equation"]
            OP3["Solver </br> Continuity Equation"]
            OP4["Solver </br> Update Turbulence"]
        end

        OP1 --> OP2
        OP2 --> OP3
        OP3 --> OP4

        %% Physics Extensions (simplified)
        subgraph AddPhysics ["Additional Physics Modules"]
            direction TB
            POROSITY["Porosity"]
            ROTATION["Rotating Reference Frame"]
            BUOYANCY["Boussinesq Approximation"]
        end
        POROSITY -.-> OP1
        ROTATION -.-> OP1
        BUOYANCY -.-> OP2
        BUOYANCY -.-> OP3

        style MAIN fill:#E3F2FD
        style AddPhysics fill:#E3F2FD
        style OP1 fill:#2196F3,color:#fff
        style OP2 fill:#FF9800,color:#fff
        style OP3 fill:#9C27B0,color:#fff
        style OP4 fill:#607D8B,color:#fff

The solver defines the main operations to be executed: momentum, continuity, and turbulence model updates.
Additional physics models, such as porosity, rotation, or buoyancy, can modify or add operations.
In this case, the porosity model and the Rotating Reference Frame add momentum source terms, while the Boussinesq model adds a temperature equation and modifies the continuity equation.

Each solver maintains a list of additional physics models that can add or modify operations.
These additional physics models are defined externally from the solver and only need to comply with the IncompressibleFluidModel interface, which can be defined separately by each solver.
The execution order is determined at runtime based on the metadata specified in each model.


.. code-block:: python

    # Pseudocode illustrating solver and model structure
    @Solver
    class IncompressibleFluidSolver:
        models: list[IncompressibleFluidModel]  # Additional physics models
        @Solver.operation(...)
        def momentum(self, ...): pass
        @Solver.operation(...)
        def continuity(self, ...): pass
        @Solver.operation(...)
        def update_turbulence(self, ...): pass

    @IncompressibleFluidModel.register
    class BoussinesqModel:
        @IncompressibleFluidModel.operation(...)
        def temperature_equation(self, ...): pass

    @IncompressibleFluidModel.register
    class PorosityModel:
        @IncompressibleFluidModel.operation(...)
        def add_momentum_source(self, ...): pass

    @IncompressibleFluidModel.register
    class RotatingReferenceFrame:
        @IncompressibleFluidModel.operation(...)
        def add_momentum_source(self, ...): pass


As a result, the solver can be easily extended with new physics using minimal additional code.
Users only need to define a new model that adds the desired operations to the solver at runtime, without modifying the core solver.
New models can be registered via the plugin system and selected in input files.

Conceptual Implementation
-------------------------

After solvers and models are initialized, their operations must be identified and sorted to determine the correct execution order.
This order is managed by the **Operations** class, shown conceptually below:

.. code-block:: python

    # Pseudocode showing how operations are stored and executed
    class Operations:
        ops: list[Operation]  # All operations to execute

        def run(self, ...): pass

    class Operation:
        func: callable
        sub_operations: list[Operation]  # Nested operations
        metadata: Any  # Metadata for sorting or description

        def run(self, ...): pass

An operation represents a single computational operation in a solver or model, functioning as a callable task.
Each solver or model can define multiple operations stored as `Operation` objects.
These can hold sub-operations and metadata to assist in sorting and dependency management.

After sorting (detailed in future documentation), the `Operations` class holds all operations required to run the newly-configured solver.

This modular design allows users to add or remove physical effects without altering the core solver structure, encouraging maintainability and reuse.

.. note::
    Implementation is still a work in progress.

Simulation with Multiple Domains/Solvers
----------------------------------------

The same concept can be extended to multi-physics scenarios with multiple domains and solvers.
Each solver defines its own operations, while coupling between solvers is managed automatically at runtime based on defined physics models and settings.

The diagram below illustrates this for two domains in a conjugate heat transfer example:

.. mermaid::

    %%{init: {'flowchart': { 'htmlLabels': true, 'wrap': true }}}%%
    flowchart TB
        subgraph setup1 ["Setup"]
            direction TB
            S1A["Initialize Fields"]
        end

        subgraph setup2 ["Setup"]
            direction TB
            S2A["Initialize Temperature Field"]
        end

        subgraph S1 ["Non-Thermal Fluid Solver"]
            direction TB
            S1C["Momentum Predictor"]
            S1C --> S1E["Solve Energy Equation"]
            S1E --> S1F["Pressure Corrector<br/>PISO Loop"]
            S1F --> S1J["Update Turbulence Model"]
        end

        subgraph S2 ["Thermal Solid Solver"]
            direction TB
            S2B["Solve Energy Equation"] --> S2D["Update Solid Properties"]
        end

        setup1 --> S1C
        setup2 --> S2B
        S1E --> S2B
        S2B --> S1E

Here, a non-thermal fluid solver and a thermal solid solver solve a conjugate heat transfer problem.
Each solver has its own initialization phase. Once the **Operations** are determined, coupling between them is established.
The operation sequences of both solvers are then updated to include inter-solver data exchange.

.. note::
    Implementation is still a work in progress.

Initialization Stage
--------------------

The initialization stage sets up the simulation environment — reading input files, initializing fields, and preparing solvers and models.

Although not yet finalized, the planned initialization stage will perform the following tasks:

1. Read and parse input files and configuration settings.
2. Initialize computational domains and meshes.
3. Exchange dependencies between operations (e.g., adding a Boussinesq approximation modifies the pressure equation formulation).
4. Define solver operations based on selected physics models.
5. Sort operations to determine correct execution order.

Plugin Architecture
-------------------

Motivation
^^^^^^^^^^

Modern scientific and engineering workflows require flexible, extensible frameworks.
FoamAdapter’s plugin architecture enables users and developers to add new physics models, boundary conditions, and solver modules without modifying core code.
This promotes maintainability, collaboration, and rapid prototyping.

Concept
^^^^^^^

FoamAdapter implements a runtime-extensible plugin/configuration system using Pydantic discriminated unions and a registry pattern.
The core idea is to allow new plugin types (e.g., models, fields, solvers) to be dynamically registered at runtime or via Python entry points (setuptools).
Each plugin type (such as a physics model or boundary condition) is managed by a registry, which collects available classes and exposes a unified configuration model for validation and schema generation.

**Background: Pydantic Discriminated Unions**

Pydantic supports discriminated unions f    or type-safe configuration, but union members must be defined at model creation time.
For example, the following ensures a pet is either a Cat, Dog, or Lizard:

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
        pet_type: Literal['lizard']
        scales: bool

    class Model(BaseModel):
        pet: Union[Cat, Dog, Lizard] = Field(discriminator='pet_type')
        n: int

This works for static unions but cannot be extended dynamically — a limitation for plugin systems.

**How FoamAdapter Solves This**

Plugins are registered using a decorator-based API.
Each registration automatically rebuilds the Pydantic model for that plugin type, updating the discriminated union to include new plugins.
Thus, the configuration model always reflects available plugins and remains valid for schema generation and input validation.

Example:

.. code-block:: python

    ShapeBase.register(TriangleConfig)
    shape = ShapeBase.plugin_model(shape={"shape_type": "triangle", "base": 3.0, "height": 4.0}, color="yellow")

This approach enables **true runtime extensibility** while maintaining strong input validation.
The updated model can always be retrieved via the `plugin_model` attribute.

Usage
^^^^^

To add a new plugin, users define a Python class and register it with the corresponding base class:

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
The unified configuration and schema system simplifies UI generation, input validation, and documentation of available plugins.

Model Introspection and Schema Generation
-----------------------------------------

Model configuration and validation in FoamAdapter are powered by Pydantic, which provides input validation and automatic JSON Schema generation.
This mechanism enables model discovery, UI integration, and automatic documentation.

Pydantic’s schema generation supports:

* Input validation — ensures consistent and type-safe configurations
* UI generation — enables dynamic form or editor creation
* Automatic documentation — exposes all field names, types, and constraints
* Metadata discovery — supports downstream analysis and automation
* AI-assisted workflows — provides structured schemas for AI-based reasoning

To retrieve a model’s JSON Schema, use:

.. code-block:: python

    # For a registered plugin or configuration model
    schema = ShapeBase.plugin_model.model_json_schema()

    # For any Pydantic model
    schema = MyModel.model_json_schema()

This unified mechanism allows programmatic discovery of fields, types, validation rules, and defaults.
All solvers, models, and plugins must therefore use Pydantic for input configuration.
