3-Stage Solver Initialization
=============================

NeoFOAM uses a **3-stage initialization process** to handle complex dependencies between solvers, models, and physics algorithms.
This ensures that configuration is loaded, dependencies are resolved, and runtime objects (fields, meshes) are built in the correct order.
To reduce the size of the solver code, the initialization logic is encapsulated in the ``StagedInit`` class and associated decorators that define each state in a separated file.

The Process: Load → Resolve → Build
-----------------------------------

.. mermaid::

    %%{init: {'sequence': {'actorMargin': 300}}}%%
    sequenceDiagram
        autonumber
        participant Solver
        participant Manager as StagedInit
        participant UserCode as User Handlers (@init...)

        Note right of Solver: Dependency Injection<br/>provides StagedInit instance
        Solver->>Manager: init.run()

        rect rgb(235, 245, 255)
            Note over Manager, UserCode: 1. LOAD STAGE
            Manager->>UserCode: Call @init.load()
            UserCode-->>Manager: Return LoadResult [List of Models]
            Note right of Manager: DATA: LoadResult passed to next stage
        end

        rect rgb(255, 245, 235)
            Note over Manager, UserCode: 2. RESOLVE STAGE
            Manager->>Manager: Create ConfigContext from LoadResult
            Manager->>UserCode: Call @init.resolve(ConfigContext)
            Note right of UserCode: Models wire dependencies<br/>using ConfigContext
            UserCode-->>Manager: (Completion)
            Note right of Manager: DATA: Verified Model Graph passed to next stage
        end

        rect rgb(235, 255, 240)
            Note over Manager, UserCode: 3. BUILD STAGE
            Manager->>UserCode: Call @init.build(mesh)
            UserCode-->>Manager: Return List[InitStep]
            Note right of Manager: DATA: Dependency Graph created from List[InitStep]

            loop For each InitStep in Order
                Manager->>Manager: Execute InitStep
                Note right of Manager: Object created & stored in Runtime Context
            end
        end

        Manager-->>Solver: Return Runtime Context

Detailed Data Flow
^^^^^^^^^^^^^^^^^^

The diagram illustrates how data transforms and moves through the system during initialization:

1.  **LOAD Stage (Data In)**:
    - **Input**: Configuration files (dictionaries).
    - **Output**: A ``LoadResult`` object containing a list of **Model Instances**.
    - These models are "empty shells" at this point—they have configuration data but no connections to other models and no fields.

2.  **RESOLVE Stage (Wiring)**:
    - **Input**: The ``LoadResult`` from stage 1.
    - **Mechanism**: The ``ConfigContext`` acts as a registry. Models are registered by name.
    - **Action**: Models query the ``ConfigContext`` to find their dependencies (e.g., ``context.get("transport")``).
    - **Result**: A fully connected graph of model instances, verified and ready for deployment.

3.  **BUILD Stage (Construction)**:
    - **Input**: The connected models and the mesh.
    - **Output**: A list of ``InitStep`` objects (recipes).
    - **Execution**: The ``Init Manager`` sorts these recipes topologically based on declared dependencies. It then executes them one by one.
    - **Runtime Context**: As each recipe executes (e.g., creating a field), its result is stored in the ``Runtime Context``. Subsequent recipes can look up these results (e.g., ``context["fields.U"]``) to build dependent objects.

Why 3 Stages?
-------------

1.  **Stage 1: LOAD**
    - Reads configuration files (dictionaries, YAML, etc.).
    - Discovers available models (e.g., Turbulence, Transport).
    - **No** interaction between models yet.
    - **No** mesh or heavy memory allocation.

2.  **Stage 2: RESOLVE**
    - Establishes connections between models.
    - Validates compatible configurations (e.g., "Is this turbulence model compatible with this solver?").
    - Adapts algorithms based on active models (e.g., switching to buoyant pressure solver if Boussinesq model is present).

3.  **Stage 3: BUILD**
    - The mesh is available.
    - Fields are created and memory is allocated.
    - Uses **Lazy Initialization** to ensure fields are created in dependency order (Mesh → U → Turbulence).

Implementing Staged Initialization
----------------------------------

The framework provides the ``StagedInit`` class. You define the logic for each stage using decorators.

.. code-block:: python

    from neofoam.framework.initialization import StagedInit, LoadResult, ConfigContext
    from neofoam.framework.initialization.lazy_init import InitStep

    # Create the initialization manager
    init = StagedInit("MySolverInit")

    @init.load
    def load_config() -> LoadResult:
        """
        Stage 1: Load configuration and models.
        """
        # Read OpenFOAM dictionaries or other config
        fv_solution = read_dictionary("system/fvSolution")

        # Instantiate model classes (lightweight, no fields yet)
        models = [TurbulenceModel(), TransportModel()]

        return LoadResult(core_models=[], optional_models=models)

    @init.resolve
    def resolve_dependencies(core_models: list, optional_models: list, config: ConfigContext):
        """
        Stage 2: Connect models.
        """
        # Example: Transport model might need to know about Turbulence
        for model in optional_models:
             model.connect(config)

    @init.build
    def build_runtime(mesh, core_models: list, optional_models: list) -> list[InitStep]:
        """
        Stage 3: Create runtime objects (Lazy Execution).
        """
        initializers = []

        # Define how to create fields
        initializers.append(
            InitStep(
                name="fields.U",
                initializer=lambda ctx: create_vector_field(mesh, "U"),
                depends_on=["mesh"]
            )
        )

        return initializers

Integration with Solver
-----------------------

The initialized ``StagedInit`` object is then injected into the solver using the ``@solver.initializer`` decorator.

.. code-block:: python

    from neofoam.framework import Solver, Context, Depends
    from typing import Annotated

    solver = Solver("MySolver")

    @solver.initializer
    def initialize(init_manager: Annotated[StagedInit, Depends(get_init_manager)]) -> Context:
        # Executes the 3-stage process and returns the simulation context
        return init_manager.run()

Lazy Initialization Graph
-------------------------

In the **BUILD** stage, we don't create objects immediately. Instead, we return ``InitStep`` descriptions. The framework builds a dependency graph and executes them in the correct topological order.

**Example Dependency Chain:**

.. code-block:: text

    Mesh (Root)
      │
      ├──> fields.U (Velocity)
      │      │
      │      └──> Turbulence Model
      │
      └──> fields.p (Pressure)

This ensures that `fields.U` exists before the Turbulence Model tries to access it, eliminating initialization order bugs.


Lazy BUILD Pattern
------------------

Starting with recent versions, the BUILD stage supports a lazy initialization pattern where methods can return ``InitStep`` objects instead of performing immediate execution. This provides several benefits:

1. **Explicit Dependencies**: Each initialization step declares its dependencies
2. **Automatic Ordering**: Dependencies are resolved using topological sort
3. **Cycle Detection**: Circular dependencies are caught early
4. **Better Testability**: Individual initialization steps can be tested in isolation

Basic Lazy Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@Solver.build`` decorator can return a list of ``InitStep`` objects:

.. code-block:: python

    from neofoam.framework import InitStep, Solver
    from neofoam.framework.initialization.helpers import field, operator, lazy, model

    class MySolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}

        @Solver.build
        def setup_runtime(self, mesh):
            """Return list of lazy initializers instead of executing immediately."""
            return [
                # Runtime and mesh (no dependencies)
                lazy("runtime", self._create_runtime),
                lazy("mesh", self._create_mesh, ["runtime"]),

                # Fields depend on mesh
                field("p", self._read_pressure_field, ["mesh"]),
                field("U", self._read_velocity_field, ["mesh"]),

                # Operators depend on fields
                operator("div_phi", self._create_divergence, ["fields.U"]),
                operator("laplacian_p", self._create_laplacian, ["fields.p"]),
            ]

        def _create_runtime(self, context):
            runtime = pyf.Time(...)
            return runtime

        def _create_mesh(self, context):
            runtime = context["runtime"]
            mesh = pyf.fvMesh(runtime)
            return mesh

        def _read_pressure_field(self, context):
            mesh = context["mesh"]
            return pyf.volScalarField.read_field("p", mesh)

When ``SolverInitializer.initialize()`` executes the BUILD stage, it:

1. Calls ``setup_runtime(mesh)`` to get the list of ``InitStep`` objects
2. Builds a dependency graph from the ``depends_on`` lists
3. Performs topological sort to determine execution order
4. Executes each initializer in order, passing the ``context`` dict
5. Stores results back in ``context`` for downstream dependencies

Helper Functions
~~~~~~~~~~~~~~~~

The framework provides helper functions to create ``InitStep`` objects with automatic naming:

.. code-block:: python

    from neofoam.framework.initialization.helpers import field, operator, lazy, model

    # field(name, initializer, dependencies) -> InitStep with name="fields.{name}"
    field("p", lambda ctx: read_field("p", ctx["mesh"]), ["mesh"])

    # operator(name, initializer, dependencies) -> InitStep with name="operators.{name}"
    operator("div_phi", lambda ctx: create_div(ctx["fields.U"]), ["fields.U"])

    # model(name, initializer, dependencies) -> InitStep with name="models.{name}"
    model("turbulence", lambda ctx: create_turbulence(...), ["fields.U", "fields.p"])

    # lazy(name, initializer, dependencies) -> InitStep with custom name
    lazy("algorithm", lambda ctx: create_algorithm(...), ["models.turbulence"])

All helpers default to an empty dependency list ``[]`` if not specified.

Dependency Resolution
~~~~~~~~~~~~~~~~~~~~~

Dependencies are specified as strings matching the ``name`` of other ``InitStep`` objects. The framework:

- Uses ``networkx.lexicographical_topological_sort`` for deterministic ordering
- Detects cycles and raises ``CyclicDependencyError`` before execution
- Ensures each initializer executes exactly once

Example dependency chain:

.. code-block:: python

    runtime (no deps)
      ↓
    mesh (depends on ["runtime"])
      ↓
    fields.p, fields.U (both depend on ["mesh"])
      ↓
    models.turbulence (depends on ["fields.U", "fields.p"])
      ↓
    algorithm (depends on ["models.turbulence"])

Execution order: ``runtime → mesh → fields.p → fields.U → models.turbulence → algorithm``

Context Passing
~~~~~~~~~~~~~~~

Each initializer receives a ``context`` dictionary containing all previously initialized objects:

.. code-block:: python

    def _create_algorithm(self, context):
        # Access dependencies via their names
        turbulence = context["models.turbulence"]
        p_field = context["fields.p"]
        U_field = context["fields.U"]

        # Create algorithm using dependencies
        algorithm = PIMPLEAlgorithm(
            turbulence=turbulence,
            pressure=p_field,
            velocity=U_field
        )
        return algorithm

The context is automatically populated as each initializer completes.

Complete Example
~~~~~~~~~~~~~~~~

Here's a full example showing lazy initialization for an incompressible solver:

.. code-block:: python

    from neofoam.framework import Solver
    from neofoam.framework.initialization.helpers import field, operator, lazy, model
    from pydantic import BaseModel

    class IncompressibleSolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}

        @Solver.build
        def setup_runtime(self, mesh):
            """Lazy initialization with explicit dependencies."""
            return [
                # Core runtime objects
                lazy("runtime", self._create_runtime),
                lazy("mesh", self._create_mesh, ["runtime"]),

                # Read fields from disk
                field("p", self._read_pressure, ["mesh"]),
                field("U", self._read_velocity, ["mesh"]),
                field("phi", self._read_flux, ["mesh"]),

                # Create physics models
                model("laminarTransport", self._create_transport, ["fields.U"]),
                model("turbulence", self._create_turbulence,
                      ["fields.U", "fields.phi", "fields.laminarTransport"]),

                # Create algorithm (needs all fields and models)
                lazy("algorithm", self._create_algorithm,
                     ["fields.p", "fields.U", "models.turbulence"]),
            ]

        def _create_runtime(self, context):
            return pyf.Time(self.argv)

        def _create_mesh(self, context):
            return pyf.fvMesh(context["runtime"])

        def _read_pressure(self, context):
            return pyf.volScalarField.read_field("p", context["mesh"])

        def _read_velocity(self, context):
            return pyf.volVectorField.read_field("U", context["mesh"])

        def _read_flux(self, context):
            return pyf.surfaceScalarField.read_field("phi", context["mesh"])

        def _create_transport(self, context):
            return pyf.singlePhaseTransportModel(
                context["fields.U"],
                context["fields.phi"]
            )

        def _create_turbulence(self, context):
            return pyf.incompressibleTurbulenceModel.New(
                context["fields.U"],
                context["fields.phi"],
                context["fields.laminarTransport"]
            )

        def _create_algorithm(self, context):
            # Access all dependencies
            mesh = context["mesh"]
            p = context["fields.p"]
            U = context["fields.U"]
            turbulence = context["models.turbulence"]

            return PIMPLEAlgorithm(mesh, p, U, turbulence)


Benefits of Lazy Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Explicit Dependencies**: Dependencies are visible in code, not implicit in execution order

.. code-block:: python

    # Clear that turbulence needs U and phi
    model("turbulence", self._create_turbulence, ["fields.U", "fields.phi"])

**Testability**: Test individual initialization steps in isolation

.. code-block:: python

    def test_turbulence_initialization():
        solver = IncompressibleSolver()

        # Mock context with only required dependencies
        context = {
            "fields.U": mock_velocity_field,
            "fields.phi": mock_flux_field,
            "fields.laminarTransport": mock_transport,
        }

        # Test individual initializer
        turbulence = solver._create_turbulence(context)
        assert turbulence is not None

**Automatic Ordering**: No need to manually order initialization calls - the framework handles it

**Early Error Detection**: Circular dependencies detected before any initialization runs


Complete Example
----------------

Here's a realistic example with multiple interdependent models:

.. code-block:: python

    from pydantic import BaseModel, Field
    from neofoam.framework import Model, Solver, ConfigContext, SolverInitializer


    class TransportModel(BaseModel):
        """Transport properties - no dependencies on other models."""

        model_config = {"arbitrary_types_allowed": True}
        name: str = "transport"
        viscosity: float = 0.0
        density: float = 0.0

        @Model.load
        def load_properties(self):
            # Load from transportProperties file
            self.viscosity = 1e-6
            self.density = 1000.0

        @Model.resolve_dependencies
        def validate(self, config: ConfigContext):
            if self.viscosity <= 0:
                raise ValueError("Invalid viscosity")

        @Model.build
        def create_fields(self, mesh):
            # Create nu and rho fields
            pass


    class TurbulenceModel(BaseModel):
        """Turbulence model - depends on transport for viscosity."""

        model_config = {"arbitrary_types_allowed": True}
        name: str = "turbulence"
        coefficients: dict = Field(default_factory=dict)
        transport_ref = None  # Set during RESOLVE_DEPENDENCIES

        @Model.load
        def load_coefficients(self):
            self.coefficients = {"C_mu": 0.09, "sigma_k": 1.0}

        @Model.resolve_dependencies
        def connect_transport(self, config: ConfigContext):
            # Get transport model for viscosity access
            self.transport_ref = config.get("transport")
            if not self.transport_ref:
                raise RuntimeError("Transport model required")

        @Model.build
        def create_fields(self, mesh):
            # Use transport viscosity for initial estimates
            nu = self.transport_ref.viscosity
            # Create k, epsilon fields...
            pass


    class PimpleSolver(BaseModel):
        """PIMPLE solver with multiple models."""

        model_config = {"arbitrary_types_allowed": True}

        transport: TransportModel = Field(default_factory=TransportModel)
        turbulence: TurbulenceModel = Field(default_factory=TurbulenceModel)

        max_iterations: int = 100
        all_models_ready: bool = False

        def get_models(self):
            """Required: Tell initializer which models we have."""
            return [self.transport, self.turbulence]

        @Solver.load
        def load_control(self):
            self.max_iterations = 100

        @Solver.resolve_dependencies
        def verify_models(self, config: ConfigContext):
            # Verify all models configured correctly
            for model in self.get_models():
                if not hasattr(model, 'transport_ref') or model.name == "transport":
                    continue
                if model.transport_ref is None:
                    raise RuntimeError(f"{model.name} missing transport reference")
            self.all_models_ready = True

        @Solver.build
        def create_context(self, mesh):
            # Set up solver runtime context
            pass


    # Usage
    mesh = load_mesh()  # Your mesh loading code
    solver = PimpleSolver()
    initializer = SolverInitializer(solver)
    initializer.initialize(mesh=mesh)

    # Now solver and all models are fully initialized
    assert solver.turbulence.transport_ref is solver.transport


Key Design Decisions
--------------------

1. **Models before Solver**: Within each stage, models initialize first. This lets the solver's RESOLVE_DEPENDENCIES method validate that all models are properly set up.

2. **Automatic Registration**: Models are registered in the ``ConfigContext`` automatically after their LOAD stage, using their ``name`` attribute.

3. **get_models() Method**: Solvers must implement ``get_models()`` to tell the initializer which models to process.

4. **Pydantic BaseModel**: Use ``model_config = {"arbitrary_types_allowed": True}`` to allow storing references to other models and non-Pydantic types like mesh objects.

5. **Stage Decorators on Classes**: Decorators are accessed via ``Model.read_files``, ``Model.configure``, ``Model.setup`` (and ``Solver.*``) to make the stage association clear in the code.


Error Handling
--------------

Raise exceptions in any stage to halt initialization:

.. code-block:: python

    @Model.resolve_dependencies
    def connect_required_model(self, config: ConfigContext):
        required = config.get("required_model")
        if required is None:
            raise RuntimeError("Required model not found in registry")
        self.required_ref = required

The ``SolverInitializer`` does not catch exceptions, allowing them to propagate for proper error handling in your application.


Adaptive Model Behavior with Configurable
--------------------------------------------

``Configurable`` enables models to expose behavior switches that other models can modify during the RESOLVE_DEPENDENCIES stage. This allows models to dynamically select different implementations or algorithm variants based on the presence of other models.

Basic Concept
~~~~~~~~~~~~~

Think of ``Configurable`` as a parameter that changes **which operations** a model returns, not just a configuration value. When another model modifies an adaptable field, it switches the model's behavior.

.. code-block:: python

    from neofoam.framework import Configurable

    # Implementations for different behaviors
    class StandardPressure:
        def get_operations(self):
            return ["momentum", "solve_pressure", "correct_velocity"]

    class BuoyantPressure:
        def get_operations(self):
            return ["momentum", "add_buoyancy", "solve_pressure_buoyant", "correct_velocity"]

    # Model with adaptable behavior
    class PressureAlgorithm(BaseModel):
        name: str = "pressure"

        # Configurable - other models can change this
        use_buoyancy: Configurable[bool] = False

        # Regular field - not adaptable
        tolerance: float = Field(default=1e-6, gt=0)

        # Dispatch to implementation
        _implementations = {
            False: StandardPressure,
            True: BuoyantPressure
        }

        def get_operations(self):
            impl = self._implementations[self.use_buoyancy]()
            return impl.get_operations()

Usage Pattern
~~~~~~~~~~~~~

Other models modify adaptable fields during RESOLVE_DEPENDENCIES:

.. code-block:: python

    class BuoyancyModel(BaseModel):
        name: str = "buoyancy"

        @Model.resolve_dependencies
        def configure(self, config: ConfigContext):
            # Get pressure algorithm
            pressure = config.get("pressure")

            # Switch it to buoyancy variant
            pressure.use_buoyancy = True  # ← Switches implementation!

Result: ``pressure.get_operations()`` now returns buoyancy operations automatically.

Multiple Adaptable Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~

Models can have multiple adaptable fields for complex dispatch:

.. code-block:: python

    class PressureVelocityCoupling(BaseModel):
        name: str = "pressure_velocity"

        # Multiple adaptable fields
        algorithm: str = Configurable(default="SIMPLE")
        use_buoyancy: bool = Configurable(default=False)

        # Tuple-based dispatch
        _implementations = {
            ("SIMPLE", False): SIMPLEStandard,
            ("SIMPLE", True): SIMPLEBuoyant,
            ("PISO", False): PISOStandard,
            ("PISO", True): PISOBuoyant,
            ("PIMPLE", False): PIMPLEStandard,
            ("PIMPLE", True): PIMPLEBuoyant,
        }

        def get_operations(self):
            key = (self.algorithm, self.use_buoyancy)
            impl = self._implementations[key]()
            return impl.get_operations()

Querying Adaptable Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``ConfigContext.get_adaptable_fields()`` to discover what's adaptable:

.. code-block:: python

    @Model.resolve_dependencies
    def configure(self, config: ConfigContext):
        # See what's adaptable
        adaptable = config.get_adaptable_fields("pressure")
        # Returns: {"use_buoyancy": False}

        # Check before modifying
        if "use_buoyancy" in adaptable:
            pressure = config.get("pressure")
            pressure.use_buoyancy = True

Multiple Model Instances
~~~~~~~~~~~~~~~~~~~~~~~~~

Models can have multiple instances (e.g., multiple heat sources):

.. code-block:: python

    class HeatSource(BaseModel):
        name: str  # "heat_source_1", "heat_source_2", etc.

        enabled: bool = Configurable(default=True)
        power: float = Field(default=1000.0, gt=0)

        def get_operations(self):
            if self.enabled:
                return [f"add_heat_{self.name}"]
            return []

    class Solver(BaseModel):
        heat_sources: list[HeatSource]

        @Solver.load
        def load_sources(self):
            self.heat_sources = [
                HeatSource(name="heat_source_1", power=1000.0),
                HeatSource(name="heat_source_2", power=500.0),
                HeatSource(name="heat_source_3", power=2000.0),
            ]

        def get_models(self):
            return self.heat_sources

Query multiple instances with ``ConfigContext`` helpers:

.. code-block:: python

    @Model.resolve_dependencies
    def configure(self, config: ConfigContext):
        # Get all heat sources by type
        sources = config.get_by_type(HeatSource)
        for source in sources:
            if source.power > 1500:
                source.enabled = False

        # Or by name prefix
        sources = config.get_by_prefix("heat_source_")
        # Returns: {"heat_source_1": ..., "heat_source_2": ..., ...}

Benefits of Configurable
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Type-Driven**: Just use ``Configurable()`` instead of ``Field()``
2. **Auto-Discovery**: ``get_adaptable_fields()`` scans field metadata
3. **Auto-Validation**: Pydantic validates all changes
4. **Clean Separation**: Each behavior variant is a separate implementation class
5. **Dynamic Selection**: Implementations chosen at runtime based on available models

Example: Pressure Algorithm Adaptation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete example showing how buoyancy model adapts pressure algorithm:

.. code-block:: python

    # Implementations
    class StandardPressure:
        def get_operations(self):
            return ["momentum", "pressure", "correct"]

    class BuoyantPressure:
        def get_operations(self):
            return ["momentum", "buoyancy_source", "pressure_buoyant", "correct"]

    # Pressure model with adaptable behavior
    class PressureModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "pressure"

        use_buoyancy: bool = Configurable(default=False)

        _implementations = {
            False: StandardPressure,
            True: BuoyantPressure
        }

        def get_operations(self):
            return self._implementations[self.use_buoyancy]().get_operations()

    # Buoyancy model that adapts pressure
    class BuoyancyModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "buoyancy"

        @Model.resolve_dependencies
        def configure(self, config: ConfigContext):
            pressure = config.get("pressure")
            if pressure:
                pressure.use_buoyancy = True

    # Solver
    class Solver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        pressure: PressureModel = Field(default_factory=PressureModel)
        buoyancy: BuoyancyModel = Field(default_factory=BuoyancyModel)

        def get_models(self):
            return [self.pressure, self.buoyancy]

    # Usage
    solver = Solver()
    initializer = SolverInitializer(solver)
    initializer.initialize(mesh)

    # Pressure automatically uses buoyancy variant
    ops = solver.pressure.get_operations()
    # Returns: ["momentum", "buoyancy_source", "pressure_buoyant", "correct"]


Testing
-------

The test suite is organized in ``test/initialization/``:

- ``test_fixtures.py`` - Shared test models
- ``test_basic.py`` - Basic initialization tests
- ``test_load_stage.py`` - LOAD stage tests
- ``test_resolve_dependencies_stage.py`` - RESOLVE_DEPENDENCIES stage tests
- ``test_build_stage.py`` - BUILD stage tests
- ``test_lazy_init.py`` - InitStep dataclass and helper function tests
- ``test_lazy_build_integration.py`` - Lazy BUILD stage integration tests
- ``test_initialization_order.py`` - Execution order tests
- ``test_error_handling.py`` - Error condition tests
- ``test_config_context.py`` - ConfigContext tests
- ``test_decorators.py`` - Decorator behavior tests
- ``test_configurable_field.py`` - Configurable behavior and dispatch tests
- ``test_incompressible_fluid.py`` - Full solver initialization tests

Run all tests:

.. code-block:: bash

    pytest test/initialization/ -v

Testing Lazy Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When testing lazy BUILD methods, verify the returned InitStep objects:

.. code-block:: python

    def test_setup_runtime_returns_lazy_init():
        solver = MySolver()
        result = solver.setup_runtime(mesh=None)

        # Verify returns list of InitStep
        assert isinstance(result, list)
        assert all(isinstance(item, InitStep) for item in result)

        # Verify expected initializers
        names = [li.name for li in result]
        assert "runtime" in names
        assert "mesh" in names
        assert "fields.p" in names

        # Verify dependencies
        for li in result:
            if li.name == "fields.p":
                assert "mesh" in li.depends_on

For integration testing, use ``SolverInitializer`` to execute the full initialization:

.. code-block:: python

    def test_full_initialization_with_lazy_build():
        solver = MySolver()
        initializer = SolverInitializer(solver)

        # Execute full initialization
        context = initializer.initialize(mesh=None)

        # Verify context contains all initialized objects
        assert "runtime" in context
        assert "mesh" in context
        assert "fields.p" in context

        # Verify solver state updated
        assert solver.setup_complete
