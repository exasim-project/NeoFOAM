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
- ``test_initialization_order.py`` - Execution order tests
- ``test_error_handling.py`` - Error condition tests
- ``test_config.py`` - ConfigContext tests
- ``test_decorators.py`` - Decorator behavior tests
- ``test_adaptable_field.py`` - Configurable behavior and dispatch tests

Run all tests:

.. code-block:: bash

    pytest test/initialization/ -v
