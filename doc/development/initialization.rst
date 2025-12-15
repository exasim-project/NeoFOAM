3-Stage Initialization
======================

The 3-stage initialization framework provides a structured approach for initializing solvers and their models. It solves the problem of complex dependencies between models while ensuring all required data is available at each step.

Why 3 Stages?
-------------

In CFD simulations, models often depend on each other. For example:

- A turbulence model needs transport properties (viscosity)
- An algorithm needs references to both turbulence and transport models
- All models need the mesh, but the mesh might not exist when configuration is loaded

A single-stage initialization can't handle these dependencies elegantly. The 3-stage approach separates concerns:

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────┐
    │ Stage 1: LOAD                                                    │
    │ ─────────────────                                                │
    │ • Load configuration from files                                  │
    │ • No dependencies between models yet                             │
    │ • Each model loads its own data independently                    │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │ Stage 2: RESOLVE_DEPENDENCIES                                    │
    │ ──────────────────                                               │
    │ • Models can reference each other via ConfigContext              │
    │ • Validate configurations                                        │
    │ • Establish inter-model dependencies                             │
    │ • Still no mesh available                                        │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │ Stage 3: BUILD                                                   │
    │ ─────────────────                                                │
    │ • Mesh is now available                                          │
    │ • Create fields, matrices, runtime structures                    │
    │ • All dependencies resolved from previous stage                  │
    └──────────────────────────────────────────────────────────────────┘


Quick Start
-----------

Here's a minimal example showing all three stages:

.. code-block:: python

    from pydantic import BaseModel, Field
    from foamadapter.framework import Model, Solver, ConfigContext, SolverInitializer

    class MyModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        
        name: str = "mymodel"
        data: dict = Field(default_factory=dict)
        other_model_ref = None
        
        @Model.load
        def load_data(self):
            """Stage 1: Load from files."""
            self.data = {"viscosity": 1e-6}
        
        @Model.resolve_dependencies
        def connect(self, config: ConfigContext):
            """Stage 2: Connect to other models."""
            self.other_model_ref = config.get("other")
        
        @Model.build
        def init_fields(self, mesh):
            """Stage 3: Create fields on mesh."""
            # self.field = create_field(mesh, self.data["viscosity"])
            pass


    class MySolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        
        mymodel: MyModel = Field(default_factory=MyModel)
        
        def get_models(self):
            return [self.mymodel]
        
        @Solver.load
        def load_config(self):
            pass
        
        @Solver.resolve_dependencies
        def validate(self, config: ConfigContext):
            pass
        
        @Solver.build
        def create_context(self, mesh):
            pass

    # Run initialization
    solver = MySolver()
    initializer = SolverInitializer(solver)
    initializer.initialize(mesh=some_mesh)


Core Components
---------------

Stage Decorators
~~~~~~~~~~~~~~~~

Each stage has a decorator accessed via ``Model.`` or ``Solver.``:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Decorator
     - When Called
     - Arguments
   * - ``@Model.load``
     - First, before any other stage
     - None (just ``self``)
   * - ``@Model.resolve_dependencies``
     - After all LOAD complete
     - ``config: ConfigContext``
   * - ``@Model.build``
     - After all RESOLVE_DEPENDENCIES complete
     - ``mesh``

The same decorators exist for solvers: ``@Solver.load``, ``@Solver.resolve_dependencies``, ``@Solver.build``.


ConfigContext
~~~~~~~~~~~~~

The ``ConfigContext`` enables inter-model communication during the RESOLVE_DEPENDENCIES stage:

.. code-block:: python

    @Model.resolve_dependencies
    def connect_dependencies(self, config: ConfigContext):
        # Get a model by name
        transport = config.get("transport")
        
        # Check if a model exists
        if config.contains("turbulence"):
            self.turbulence = config.get("turbulence")
        
        # Get all registered models
        all_models = config.all()  # Returns dict[str, Model]

Models are automatically registered using their ``name`` attribute after their LOAD stage completes.


SolverInitializer
~~~~~~~~~~~~~~~~~

The ``SolverInitializer`` orchestrates the entire initialization:

.. code-block:: python

    from foamadapter.framework import SolverInitializer

    solver = MySolver()
    initializer = SolverInitializer(solver)
    
    # Option 1: Full initialization in one call
    initialized_solver = initializer.initialize(mesh=my_mesh)
    
    # Option 2: Access the registry after initialization
    initializer.initialize(mesh=my_mesh)
    all_models = initializer.config.all()


Execution Order
---------------

Within each stage, **models are initialized before the solver**:

.. code-block:: text

    LOAD stage:
        1. model1.load_method()
        2. model2.load_method()
        3. model3.load_method()
        4. solver.load_method()  ← Solver last

    RESOLVE_DEPENDENCIES stage:
        1. model1.resolve_dependencies_method(config)
        2. model2.resolve_dependencies_method(config)
        3. model3.resolve_dependencies_method(config)
        4. solver.resolve_dependencies_method(config)  ← Solver can validate all models

    BUILD stage:
        1. model1.build_method(mesh)
        2. model2.build_method(mesh)
        3. model3.build_method(mesh)
        4. solver.build_method(mesh)

This allows the solver's CONFIGURE method to verify all models are properly configured.


Lazy BUILD Pattern
------------------

Starting with recent versions, the BUILD stage supports a lazy initialization pattern where methods can return ``LazyInit`` objects instead of performing immediate execution. This provides several benefits:

1. **Explicit Dependencies**: Each initialization step declares its dependencies
2. **Automatic Ordering**: Dependencies are resolved using topological sort
3. **Cycle Detection**: Circular dependencies are caught early
4. **Better Testability**: Individual initialization steps can be tested in isolation

Basic Lazy Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@Solver.build`` decorator can return a list of ``LazyInit`` objects:

.. code-block:: python

    from foamadapter.framework import LazyInit, Solver
    from foamadapter.framework.initialization.helpers import field, operator, lazy, model

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

1. Calls ``setup_runtime(mesh)`` to get the list of ``LazyInit`` objects
2. Builds a dependency graph from the ``depends_on`` lists
3. Performs topological sort to determine execution order
4. Executes each initializer in order, passing the ``context`` dict
5. Stores results back in ``context`` for downstream dependencies

Helper Functions
~~~~~~~~~~~~~~~~

The framework provides helper functions to create ``LazyInit`` objects with automatic naming:

.. code-block:: python

    from foamadapter.framework.initialization.helpers import field, operator, lazy, model

    # field(name, initializer, dependencies) -> LazyInit with name="fields.{name}"
    field("p", lambda ctx: read_field("p", ctx["mesh"]), ["mesh"])
    
    # operator(name, initializer, dependencies) -> LazyInit with name="operators.{name}"
    operator("div_phi", lambda ctx: create_div(ctx["fields.U"]), ["fields.U"])
    
    # model(name, initializer, dependencies) -> LazyInit with name="models.{name}"
    model("turbulence", lambda ctx: create_turbulence(...), ["fields.U", "fields.p"])
    
    # lazy(name, initializer, dependencies) -> LazyInit with custom name
    lazy("algorithm", lambda ctx: create_algorithm(...), ["models.turbulence"])

All helpers default to an empty dependency list ``[]`` if not specified.

Dependency Resolution
~~~~~~~~~~~~~~~~~~~~~

Dependencies are specified as strings matching the ``name`` of other ``LazyInit`` objects. The framework:

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

    from foamadapter.framework import Solver
    from foamadapter.framework.initialization.helpers import field, operator, lazy, model
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

Migration from Immediate Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have existing BUILD methods using the builder pattern:

**Old pattern (immediate execution):**

.. code-block:: python

    @Solver.build
    def setup_runtime(self, mesh, builder):
        runtime = pyf.Time(self.argv)
        builder.set_runtime(runtime)
        
        mesh = pyf.fvMesh(runtime)
        builder.set_mesh(mesh)
        
        p = pyf.volScalarField.read_field("p", mesh)
        builder.add_field("p", p)

**New pattern (lazy initialization):**

.. code-block:: python

    @Solver.build
    def setup_runtime(self, mesh):
        return [
            lazy("runtime", lambda ctx: pyf.Time(self.argv)),
            lazy("mesh", lambda ctx: pyf.fvMesh(ctx["runtime"]), ["runtime"]),
            field("p", lambda ctx: pyf.volScalarField.read_field("p", ctx["mesh"]), ["mesh"]),
        ]

Key differences:

1. Remove ``builder`` parameter - method now takes only ``mesh``
2. Return list of ``LazyInit`` objects instead of calling builder methods
3. Declare dependencies explicitly via ``depends_on`` parameter
4. Use lambda functions or bound methods for deferred execution
5. Access dependencies via ``context`` dict in initializers

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
    from foamadapter.framework import Model, Solver, ConfigContext, SolverInitializer


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

    from foamadapter.framework import Configurable

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
        
        @Solver.read_files
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
- ``test_lazy_init.py`` - LazyInit dataclass and helper function tests
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

When testing lazy BUILD methods, verify the returned LazyInit objects:

.. code-block:: python

    def test_setup_runtime_returns_lazy_init():
        solver = MySolver()
        result = solver.setup_runtime(mesh=None)
        
        # Verify returns list of LazyInit
        assert isinstance(result, list)
        assert all(isinstance(item, LazyInit) for item in result)
        
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
