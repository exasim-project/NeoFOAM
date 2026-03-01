Solver Framework
================

NeoFOAM's solver framework uses a **Spec/Runtime architecture** combined with a **3-stage initialization process** to handle complex dependencies between solvers, models, and physics algorithms.

Architecture Overview
---------------------

The framework separates **definition** from **execution** using two core concepts:

- **Spec** (``ModelSpec``, ``SolverSpec``): Immutable definition created at module import time. Decorators register behavior (load, resolve, build, operations) without executing anything.
- **Runtime** (``ModelRuntime``, ``SolverRuntime``): Mutable per-instance state created from a Spec. Holds loaded config, resolved dependencies, and runtime objects.

.. code-block:: text

    Module Import Time              Solver Startup
    ─────────────────              ──────────────
    ModelSpec (definition)  ──►  ModelRuntime (instance)
    SolverSpec (definition) ──►  SolverRuntime (instance)

This separation allows multiple independent runtime instances from a single spec, clean testability, and plugin-based model discovery.


ModelSpec & ModelRuntime
------------------------

A ``ModelSpec`` defines a physics model (e.g., turbulence, transport). Create one using the ``Model()`` factory function and register behavior with decorators.

Creating a ModelSpec
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from neofoam.framework.model import Model
    from neofoam.framework.initialization import ConfigContext, InitStep
    from neofoam.framework.initialization.helpers import field
    from neofoam.io import BaseConfig, IOStrategy, YAML

    # Create the spec at module level
    transport = Model("Transport")

    # Register a config class
    @transport.config
    @IOStrategy(YAML("transportProperties"))
    class TransportConfig(BaseConfig):
        viscosity: float = 1e-6
        density: float = 1000.0

    # Stage 1: LOAD — return a config object (no side effects)
    @transport.load
    def load(case_dir: Path, _entry: Any) -> TransportConfig:
        return TransportConfig.load(case_dir=case_dir)

    # Stage 1b: DETECT — should this model be active?
    @transport.detect
    def detect(case_dir: Path) -> bool:
        return (case_dir / "transportProperties.yaml").exists()

    # Stage 2: RESOLVE — wire inter-model dependencies
    @transport.resolve
    def resolve(cfg: TransportConfig, ctx: ConfigContext) -> TransportConfig:
        if cfg.viscosity <= 0:
            raise ValueError("Invalid viscosity")
        return cfg

    # Stage 3: BUILD — return lazy initializers
    @transport.build
    def build(cfg: TransportConfig, _runtime: Any) -> list[InitStep]:
        return [
            field("nu", create=lambda ctx: cfg.viscosity, depends_on=["mesh"]),
        ]

Decorator Summary
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Decorator
     - Signature
     - Purpose
   * - ``@spec.config``
     - ``class MyConfig(BaseConfig)``
     - Register config class for auto-construction
   * - ``@spec.load``
     - ``(case_dir: Path, entry: Any) -> Config``
     - Load config from files (LOAD stage)
   * - ``@spec.detect``
     - ``(case_dir: Path) -> bool | list[str]``
     - Check if model should be active
   * - ``@spec.resolve``
     - ``(config, ctx: ConfigContext) -> Config``
     - Wire dependencies (RESOLVE stage)
   * - ``@spec.build``
     - ``(config, runtime) -> list[InitStep]``
     - Create lazy initializers (BUILD stage)
   * - ``@spec.operation``
     - ``(self, field1: float, cfg: Config) -> FieldUpdates``
     - Register a runtime operation
   * - ``@spec.operation_collection``
     - ``(self) -> Operations``
     - Conditional operation dispatch

ModelRuntime
~~~~~~~~~~~~

A ``ModelRuntime`` is created from a spec via ``instantiate()``:

.. code-block:: python

    # Create a runtime instance
    runtime = transport.instantiate(case_dir=Path("./case"))

    # Runtime holds the loaded config
    print(runtime.config.viscosity)  # 1e-6

    # Execute stages
    config_ctx = ConfigContext()
    runtime.run_resolve(config_ctx)      # Stage 2
    init_steps = runtime.run_build()     # Stage 3

    # Access operations
    ops = runtime.operations  # list[Operation]


SolverSpec & SolverRuntime
--------------------------

A ``SolverSpec`` defines a solver's initialization, execution graph, and operations.

Creating a SolverSpec
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from typing import Any, Annotated
    from neofoam.framework.solver import Solver
    from neofoam.framework.context import Context, FieldUpdates
    from neofoam.framework.initialization import StagedInit, Depends
    from neofoam.framework.operations import (
        Operation, Operations, StepBuilder,
        SequentialOp, IterativeOp, OperationMetadata,
    )

    solver_spec = Solver("PimpleSolver")

    # Initialization: orchestrates the 3-stage process
    @solver_spec.initializer
    def initialize(
        self: Any,
        init: Annotated[StagedInit, Depends(create_init_manager)],
    ) -> Context:
        return init.run()

    # Execution graph: defines the solver's loop structure
    @solver_spec.execution_graph_step
    def execution_graph(self: Any) -> tuple[StepBuilder, Operations]:
        builder = StepBuilder()
        # Add solver steps and loops (see Operations section)
        return builder, model_operations

    # Solver operations
    @solver_spec.operation(operation_number="1.0", name="solve_momentum")
    def solve_momentum(self: Any, U: float) -> FieldUpdates:
        return FieldUpdates({"U": U * 0.99})

Using the SolverRuntime
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create a runtime
    solver = solver_spec.instantiate(argv=["--case", "./cavity"])

    # Run initialization (executes LOAD → RESOLVE → BUILD)
    ctx = solver.initialize()

    # Build execution graph
    builder, model_ops = solver.execution_graph()

    # Access solver operations
    ops = solver.operations  # Operations container


3-Stage Initialization
----------------------

The initialization process ensures configuration is loaded, dependencies are resolved, and runtime objects are built in the correct order.

The Process: Load |rarr| Resolve |rarr| Build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |rarr| unicode:: U+2192

.. mermaid::

    %%{init: {'sequence': {'actorMargin': 300}}}%%
    sequenceDiagram
        autonumber
        participant SR as SolverRuntime
        participant SI as StagedInit
        participant MS as ModelSpec (@spec decorators)
        participant MR as ModelRuntime

        SR->>SI: init.run()

        rect rgb(235, 245, 255)
            Note over SI, MS: 1. LOAD STAGE
            SI->>MS: Detect models + load configs
            MS-->>MR: spec.instantiate() creates ModelRuntime
            MR-->>SI: Return LoadResult [list of ModelRuntime]
        end

        rect rgb(255, 245, 235)
            Note over SI, MR: 2. RESOLVE STAGE
            SI->>SI: Create ConfigContext, register models
            SI->>MR: runtime.run_resolve(ConfigContext)
            Note right of MR: Models wire dependencies,<br/>update configs
            MR-->>SI: (Completion)
        end

        rect rgb(235, 255, 240)
            Note over SI, MR: 3. BUILD STAGE
            SI->>MR: runtime.run_build()
            MR-->>SI: Return list[InitStep]
            Note right of SI: Topological sort of all InitSteps

            loop For each InitStep in order
                SI->>SI: Execute InitStep
                Note right of SI: Object stored in Context
            end
        end

        SI-->>SR: Return Context

Stage Details
^^^^^^^^^^^^^

1. **LOAD Stage**: Read configuration files, detect active models, create ``ModelRuntime`` instances. No interaction between models yet. No mesh or heavy memory allocation.

2. **RESOLVE Stage**: Models wire dependencies via ``ConfigContext``. Validate compatible configurations. Adapt algorithms based on active models.

3. **BUILD Stage**: Return ``InitStep`` objects describing how to create runtime objects. The framework sorts them topologically and executes them in dependency order, building a ``Context``.

StagedInit
~~~~~~~~~~

The ``StagedInit`` class orchestrates the 3-stage process. It is typically injected into the solver's ``@initializer`` via ``Depends``.

.. code-block:: python

    from neofoam.framework.initialization import (
        StagedInit, LoadResult, ConfigContext, InitializerBuilder,
    )

    init = StagedInit("MySolver")

    @init.load
    def load_config() -> LoadResult:
        # Detect and instantiate models
        optional_models = MyModelInterface.detect_specs_with_manifest(
            case_dir=Path("./case"),
            manifest_path=Path("./case/models.yaml"),
        )
        return LoadResult(
            core_models=[algorithm_config],
            optional_models=optional_models,
        )

    @init.resolve
    def resolve_dependencies(config: ConfigContext) -> None:
        for runtime in init.optional_models:
            runtime.run_resolve(config)

    @init.build
    def build_runtime(
        core_models: list, optional_models: list
    ) -> list[InitStep]:
        builder = InitializerBuilder()
        builder.add_resource("mesh", mesh_data)
        builder.add_core_models([("algorithm", algo_config)])
        builder.add_field("p", depends_on=["mesh"], value=0.0)
        builder.add_field("U", depends_on=["mesh"], value=0.0)

        # Each model contributes its own InitSteps
        for runtime in optional_models:
            builder.extend(runtime.run_build())

        return builder.build()

    # Execute all stages at once
    ctx = init.run()  # Returns Context

    # Or execute stages individually
    load_result = init.run_load()
    init.run_resolve(ConfigContext())
    init_steps = init.run_build()


Lazy Initialization
-------------------

In the BUILD stage, objects are not created immediately. Instead, ``InitStep`` objects describe **what** to create and **what dependencies** are needed. The framework builds a dependency graph and executes them in topological order.

InitStep
~~~~~~~~

.. code-block:: python

    from neofoam.framework.initialization import InitStep

    InitStep(
        name="fields.U",                                    # Unique identifier
        depends_on=["mesh"],                                 # Must exist before this runs
        initializer=lambda results: create_field(results["mesh"]),  # Receives dict of prior results
        category="fields",                                   # Route to ctx.fields
    )

Categories determine where results are stored in the ``Context``:

- ``"fields"`` |rarr| ``ctx.fields["U"]`` (name prefix ``fields.`` stripped)
- ``"models"`` |rarr| ``ctx.models["turbulence"]``
- ``"operators"`` |rarr| ``ctx.models["div_phi"]``
- ``"resource"`` |rarr| ``ctx.fields["mesh"]`` or ``ctx.runtime`` (top-level)

Helper Functions
~~~~~~~~~~~~~~~~

Helpers create ``InitStep`` objects with automatic naming and category:

.. code-block:: python

    from neofoam.framework.initialization import field, operator, lazy, model

    # field("p", ...) -> InitStep(name="fields.p", category="fields")
    field("p", create=lambda ctx: 0.0, depends_on=["mesh"])

    # operator("div_phi", ...) -> InitStep(name="operators.div_phi", category="operators")
    operator("div_phi", create=lambda ctx: make_div(ctx["fields.U"]), depends_on=["fields.U"])

    # model("turbulence", ...) -> InitStep(name="models.turbulence", category="models")
    model("turbulence", create=lambda ctx: make_turb(), depends_on=["fields.U", "fields.p"])

    # lazy("mesh", ...) -> InitStep(name="mesh", category="resource")
    lazy("mesh", create=lambda ctx: load_mesh())

All helpers default to an empty dependency list ``[]`` if not specified.

InitializerBuilder
~~~~~~~~~~~~~~~~~~

The fluent ``InitializerBuilder`` simplifies constructing lists of ``InitStep`` objects:

.. code-block:: python

    from neofoam.framework.initialization import InitializerBuilder

    builder = InitializerBuilder()
    builder.add_resource("mesh", mesh_data)
    builder.add_field("p", depends_on=["mesh"], value=0.0)
    builder.add_field("U", depends_on=["mesh"], value=0.0)
    builder.add_operator("div_phi", depends_on=["fields.U"], value=make_div)
    builder.add_core_models([("algorithm", algo)])

    # Extend with model-contributed steps
    builder.extend(model_runtime.run_build())

    init_steps = builder.build()  # Returns list[InitStep]

Dependency Resolution
~~~~~~~~~~~~~~~~~~~~~

Dependencies are strings matching the ``name`` of other ``InitStep`` objects. The framework:

- Uses ``networkx.lexicographical_topological_sort`` for deterministic ordering
- Detects cycles and raises ``InitializationGraphError`` before execution
- Ensures each initializer executes exactly once

Example dependency chain:

.. code-block:: text

    mesh (no deps)
      |
      +---> fields.p (depends on ["mesh"])
      |
      +---> fields.U (depends on ["mesh"])
                |
                +---> models.turbulence (depends on ["fields.U", "fields.p"])
                          |
                          +---> algorithm (depends on ["models.turbulence"])


Model Discovery & Manifests
----------------------------

Models can be discovered automatically via ``@detect`` or loaded explicitly from YAML manifests.

Auto-Detection with @detect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@detect`` decorator determines whether a model should be active. It can return a simple ``bool`` or a ``list[str]`` of instance IDs for multi-instance models.

.. code-block:: python

    # Simple detection
    @transport.detect
    def detect(case_dir: Path) -> bool:
        return (case_dir / "transportProperties.yaml").exists()

    # Multi-instance detection
    @multi_model.detect
    def detect(case_dir: Path) -> list[str]:
        with open(case_dir / "sources.yaml") as f:
            data = yaml.safe_load(f)
        return list(data.keys())  # ["instance_a", "instance_b"]

Usage:

.. code-block:: python

    result = transport.run_detect(case_dir=Path("./case"))
    # DetectResult(detected=True, instance_ids=[])

    result = multi_model.run_detect(case_dir=Path("./case"))
    # DetectResult(detected=True, instance_ids=["instance_a", "instance_b"])

YAML Manifests
~~~~~~~~~~~~~~

Manifests provide explicit model instantiation with inline config. The manifest format:

.. code-block:: yaml

    # models.yaml
    - type: MultiModel
      name: instance_a
      scale: 1.5
      offset: 0.1

    - type: MultiModel
      name: instance_b
      scale: 2.0
      offset: 0.5

Load manifests with ``load_manifest()``:

.. code-block:: python

    from neofoam.framework.model import load_manifest

    runtimes = load_manifest(
        manifest_path=Path("./case/models.yaml"),
        case_dir=Path("./case"),
        registry_name="MyModelInterface",
    )
    # Returns list[ModelRuntime], one per manifest entry

For manifest loading to work, the model must be registered with a plugin interface:

.. code-block:: python

    model_spec = Model("MultiModel").register_with(MyModelInterface)

    @model_spec.config
    class MultiModelConfig(BaseConfig):
        scale: float = 1.0
        offset: float = 0.0

Multi-Instance Build
~~~~~~~~~~~~~~~~~~~~

``@build`` always receives ``runtime`` as its second parameter. Use it to access the instance name for multi-instance models:

.. code-block:: python

    @multi_model.build
    def build(cfg: MultiModelConfig, runtime: Any) -> list[InitStep]:
        # Use runtime.name for instance-specific field names
        field_name = f"model_field_{runtime.name}"
        return [
            field(field_name, create=lambda ctx: cfg.scale, depends_on=["mesh"]),
        ]

Combined Discovery
~~~~~~~~~~~~~~~~~~

A common pattern combines manifest loading with auto-detection for models not covered by the manifest:

.. code-block:: python

    class MyModelInterface:
        @classmethod
        def detect_specs_with_manifest(cls, case_dir, manifest_path):
            # 1. Load models from manifest
            manifest_runtimes = load_manifest(manifest_path, case_dir, "MyModelInterface")

            # 2. Auto-detect remaining models
            for spec, result in cls.detect_specs(case_dir=case_dir):
                if result.detected:
                    # Skip if already in manifest
                    ...

            return combined_runtimes


Operations & Execution Graph
-----------------------------

Operations define the computational steps a solver or model performs at runtime. They are registered with decorators and composed into a nested execution graph.

Defining Operations
~~~~~~~~~~~~~~~~~~~

Operations are decorated functions that receive field values from the ``Context`` and return ``FieldUpdates`` to modify them:

.. code-block:: python

    from neofoam.framework.context import FieldUpdates

    @solver_spec.operation(operation_number="1.0", name="solve_momentum")
    def solve_momentum(self: Any, U: float, p: float) -> FieldUpdates:
        # Parameter names match ctx.fields keys (auto-injected)
        new_U = U - 0.01 * p
        return FieldUpdates({"U": new_U})

    @model_spec.operation(
        operation_number="2.5",
        depends_on=["solve_momentum"],  # Execute after this operation
        name="update_turbulence",
    )
    def update_turbulence(
        self: Any,
        U: float,                       # Auto-injected from ctx.fields["U"]
        cfg: TurbulenceConfig,          # Auto-injected by type from runtime.config
    ) -> FieldUpdates:
        return FieldUpdates({"k": U * cfg.c_mu})

    # Operations can also receive the full Context via a `ctx` parameter
    @model_spec.operation(operation_number="3.0", name="coupled_step")
    def coupled_step(
        ctx: Any,                       # Full Context object
        model_field: float,             # Also auto-injected from ctx.fields
        cfg: TurbulenceConfig,
    ) -> FieldUpdates:
        mesh = ctx.mesh
        return FieldUpdates({"model_field": model_field * 0.9})

Key features of operation auto-injection:

- **Field injection**: Parameter names matching ``ctx.fields`` keys are automatically populated
- **Config injection**: Parameters whose **type annotation** is a ``BaseConfig`` subclass are discovered and injected from ``runtime.config`` (matched by type, not by parameter name)
- **Context injection**: A parameter named ``ctx`` receives the full ``Context`` object
- **Self binding**: ``self`` is bound to the ``ModelRuntime`` or ``SolverRuntime`` instance
- **Lazy state**: Use ``if not hasattr(self, '_counter'): self._counter = 0`` for per-runtime state

Conditional Operation Dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``@operation_collection`` to select different operations based on runtime config:

.. code-block:: python

    @model_spec.operation_collection
    def collected_operations(self: Any) -> Operations:
        if self.config.coupled:
            # Return coupled variant
            op = Operation(
                func=SequentialOp(coupled_step_fn),
                metadata=OperationMetadata(
                    op_name="coupled_step",
                    operation_number=OperationNumber("2.9"),
                ),
            )
        else:
            # Return standalone variant
            op = Operation(
                func=SequentialOp(standalone_step_fn),
                metadata=OperationMetadata(
                    op_name="standalone_step",
                    operation_number=OperationNumber("2.9"),
                ),
            )
        return Operations([op])

Operation Types
~~~~~~~~~~~~~~~

Operations are wrapped in type markers that determine execution behavior:

- ``SequentialOp(func)``: Executes once per call
- ``IterativeOp(func)``: Loops while ``func`` returns ``True``
- ``ConditionalOp(func)``: Conditional execution gate (returns ``bool``)

.. code-block:: python

    from neofoam.framework.operations import (
        Operation, SequentialOp, IterativeOp, ConditionalOp, OperationMetadata,
    )

    # A step that runs once
    step = Operation(
        func=SequentialOp(lambda ctx: solve(ctx)),
        metadata=OperationMetadata(op_name="solve", operation_number=OperationNumber("1.0")),
    )

    # A loop that repeats until convergence
    loop = Operation(
        func=IterativeOp(lambda ctx: not converged(ctx)),
        metadata=OperationMetadata(op_name="outer_loop"),
    )

StepBuilder & Execution Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``StepBuilder`` creates nested loop structures for solver execution:

.. code-block:: python

    from neofoam.framework.operations import StepBuilder

    builder = StepBuilder()

    # Add a sequential step at top level
    builder.step(momentum_op)

    # Create a loop scope — returns a new StepBuilder for the loop body
    inner = builder.loop(convergence_check_op)  # -> StepBuilder
    inner.step(pressure_op)
    inner.step(correction_op)

The ``DAGResolver`` merges the solver's structural graph with model operations:

.. code-block:: python

    from neofoam.framework.graph.dag_resolver import DAGResolver

    builder, model_ops = solver.execution_graph()
    resolver = DAGResolver()
    resolved = resolver.resolve(builder, model_ops)  # -> StepBuilder

The resolver:

1. Collects all operations from the builder and model operations
2. Infers which scope each model operation belongs to (based on ``depends_on`` / ``before``)
3. Builds a global dependency graph
4. Topologically sorts within each scope
5. Rebuilds the ``StepBuilder`` with sorted operations


Context & FieldUpdates
-----------------------

The ``Context`` is the central data structure passed through initialization and operations.

.. code-block:: python

    from neofoam.framework.context import Context, FieldUpdates

    # Context after initialization
    ctx = Context(
        fields={"U": 0.0, "p": 0.0, "k": 0.0},
        models={"turbulence": turb_model, "optional_models": [rt1, rt2]},
        mesh=mesh_object,
        runtime=time_object,
    )

    # Operations return FieldUpdates to modify ctx.fields
    def my_operation(self, U: float) -> FieldUpdates:
        return FieldUpdates({"U": U * 0.99})
    # After execution: ctx.fields["U"] is updated automatically


Dependency Injection
--------------------

The framework uses ``Depends`` markers for dependency injection, primarily in solver initialization:

.. code-block:: python

    from typing import Annotated
    from neofoam.framework.initialization import Depends, StagedInit

    @solver_spec.initializer
    def initialize(
        self: Any,
        init: Annotated[StagedInit, Depends(create_init_manager)],
    ) -> Context:
        return init.run()

``Depends`` supports:

- **Callable providers**: ``Depends(factory_function)`` — calls the function to produce the value
- **String paths**: ``Depends("fields.U")`` — looks up a value in the context
- **Scoping**: ``Depends(fn, scope="time_step")`` — controls cache lifetime (``time_step``, ``iteration``, ``operation``)
- **Caching**: ``Depends(fn, cache=True)`` — caches the result within scope (default)


Complete Example
----------------

Here is a complete example showing a transport model and solver working together.

Transport Model
~~~~~~~~~~~~~~~

.. code-block:: python

    # my_solver/models/transport.py
    from pathlib import Path
    from neofoam.framework.model import Model
    from neofoam.framework.initialization import ConfigContext, InitStep
    from neofoam.framework.initialization.helpers import field
    from neofoam.framework.context import FieldUpdates
    from neofoam.io import BaseConfig, IOStrategy, YAML

    transport = Model("Transport").register_with(PhysicsModelInterface)

    @transport.config
    @IOStrategy(YAML("transportProperties"))
    class TransportConfig(BaseConfig):
        viscosity: float = 1e-6
        density: float = 1000.0

    @transport.load
    def load(case_dir: Path, _entry: Any) -> TransportConfig:
        return TransportConfig.load(case_dir=case_dir)

    @transport.detect
    def detect(case_dir: Path) -> bool:
        return (case_dir / "transportProperties.yaml").exists()

    @transport.resolve
    def resolve(cfg: TransportConfig, ctx: ConfigContext) -> TransportConfig:
        if cfg.viscosity <= 0:
            raise ValueError("Invalid viscosity")
        return cfg

    @transport.build
    def build(cfg: TransportConfig, _runtime: Any) -> list[InitStep]:
        return [
            field("nu", create=lambda ctx: cfg.viscosity, depends_on=["mesh"]),
        ]

    @transport.operation(operation_number="2.5", depends_on=["solve_momentum"])
    def update_viscosity(
        self: Any, nu: float, cfg: TransportConfig
    ) -> FieldUpdates:
        return FieldUpdates({"nu": cfg.viscosity})

Solver
~~~~~~

.. code-block:: python

    # my_solver/solver.py
    from typing import Any, Annotated
    from neofoam.framework.solver import Solver
    from neofoam.framework.context import Context, FieldUpdates
    from neofoam.framework.initialization import StagedInit, Depends
    from neofoam.framework.operations import (
        Operation, Operations, StepBuilder,
        SequentialOp, IterativeOp, OperationMetadata,
    )
    from neofoam.framework.types import OperationNumber

    solver_spec = Solver("SimpleSolver")

    @solver_spec.initializer
    def initialize(
        self: Any,
        init: Annotated[StagedInit, Depends(create_init)],
    ) -> Context:
        return init.run()

    @solver_spec.execution_graph_step
    def execution_graph(self: Any) -> tuple[StepBuilder, Operations]:
        builder = StepBuilder()

        # Outer loop
        outer = builder.loop(Operation(
            func=IterativeOp(lambda ctx: ctx.fields["iteration"] < 100),
            metadata=OperationMetadata(op_name="outer_loop"),
        ))

        # Solver step inside the loop
        outer.step(Operation(
            func=SequentialOp(lambda ctx: None),
            metadata=OperationMetadata(
                op_name="solve_momentum",
                operation_number=OperationNumber("1.0"),
            ),
        ))

        # Collect model operations
        model_ops = Operations()
        for rt in self.state.optional_models:
            model_ops.add(rt.operations)

        return builder, model_ops

    @solver_spec.operation(operation_number="1.0", name="solve_momentum")
    def solve_momentum(self: Any, U: float, p: float) -> FieldUpdates:
        return FieldUpdates({"U": U - 0.01 * p})

Running the Solver
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from neofoam.framework.graph.dag_resolver import DAGResolver

    # 1. Create runtime
    solver = solver_spec.instantiate()

    # 2. Initialize (LOAD → RESOLVE → BUILD)
    ctx = solver.initialize()

    # 3. Build execution graph and resolve dependencies
    builder, model_ops = solver.execution_graph()
    resolver = DAGResolver()
    resolved = resolver.resolve(builder, model_ops)

    # 4. Run the solver loop
    # (iterate over resolved operations, calling op.run(ctx))


Error Handling
--------------

Raise exceptions in any stage to halt initialization:

.. code-block:: python

    @model_spec.resolve
    def resolve(cfg: MyConfig, ctx: ConfigContext) -> MyConfig:
        required = ctx.get("transport")
        if required is None:
            raise RuntimeError("Transport model required but not found")
        return cfg

Exceptions propagate without being caught, allowing proper error handling in your application.

The initialization system also detects structural errors early:

- **Cyclic dependencies**: ``InitializationGraphError`` raised before any ``InitStep`` executes
- **Missing dependencies**: Detected during graph validation
- **Duplicate names**: Caught when building the dependency graph


Testing
-------

The test suite is organized across multiple directories:

**Framework component tests** (``test/framework/components/``):

- ``test_model_spec.py`` — ModelSpec decorator and instantiation tests
- ``test_operations.py`` — Operation, Operations, StepBuilder tests
- ``test_dag_resolver.py`` — DAG resolution and topological sort tests
- ``test_manifest.py`` — Manifest loading tests
- ``test_conditions.py`` — ConditionalOp tests
- ``test_graph_module.py`` — Graph validation and sorting
- ``test_operation_metadata.py`` — OperationMetadata tests
- ``test_step_number.py`` — OperationNumber tests
- ``test_visualize_dag.py`` — DAG visualization tests
- ``decorator/test_decorator.py`` — Decorator behavior tests

**Integration tests** (``test/framework/integration/dummy_solver/``):

- ``test_dummy_solver.py`` — Full solver lifecycle (init, graph, run)
- ``test_model3.py`` — Conditional operation dispatch
- ``test_model4.py`` — Multi-instance model detection and manifests
- ``test_model_registration.py`` — Plugin registration and discovery
- ``test_staged_init.py`` — StagedInit orchestration tests
- ``test_validation.py`` — Config validation tests

**Initialization tests** (``test/initialization/``):

- ``test_execution.py`` — InitStep execution and topological sort
- ``test_config_context.py`` — ConfigContext registry tests
- ``test_depends.py`` — Depends marker and dependency resolution
- ``test_helpers.py`` — InitStep helper functions and InitializerBuilder
- ``test_lazy_init.py`` — InitStep dataclass tests
- ``test_staged_init.py`` — StagedInit stage execution tests

Run all framework tests:

.. code-block:: bash

    pytest test/framework/ test/initialization/ -v

Testing ModelSpec
~~~~~~~~~~~~~~~~~

Test individual stages in isolation:

.. code-block:: python

    from neofoam.framework.model.runtime import ModelRuntime

    def test_model_build():
        rt = ModelRuntime(
            spec=transport,
            name="transport",
            config=TransportConfig(viscosity=1e-6, density=1000.0),
        )

        # Test BUILD stage
        steps = rt.run_build()
        assert isinstance(steps, list)
        names = [s.name for s in steps]
        assert "fields.nu" in names

    def test_model_operations():
        rt = ModelRuntime(
            spec=transport,
            name="transport",
            config=TransportConfig(viscosity=1e-6),
        )

        ops = rt.operations
        assert len(ops) == 1
        assert ops[0].operation_name == "update_viscosity"

Testing Full Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def test_full_initialization():
        solver = solver_spec.instantiate()
        ctx = solver.initialize()

        # Verify context
        assert "U" in ctx.fields
        assert "p" in ctx.fields
        assert ctx.mesh is not None

    def test_execution_graph():
        solver = solver_spec.instantiate()
        ctx = solver.initialize()

        builder, model_ops = solver.execution_graph()
        resolver = DAGResolver()
        resolved = resolver.resolve(builder, model_ops)

        # Verify operations are properly ordered
        # ...


Key Design Decisions
--------------------

1. **Spec/Runtime separation**: Specs are immutable definitions created at module import time. Runtimes are mutable per-instance state. This enables multiple independent instances from one definition and clean plugin registration.

2. **Config as return value**: ``@load`` returns a config object; ``@resolve`` receives and returns config. No self-mutation — configs are explicit data flowing through stages.

3. **Manifest-based discovery**: Models are discovered via YAML manifests and ``@detect`` predicates, replacing the old ``get_models()`` pattern. This decouples model registration from solver code.

4. **Operation auto-injection**: Operation parameter names are matched against ``ctx.fields`` keys. Parameters whose **type** is a ``BaseConfig`` subclass are found in ``runtime.config`` by type. A parameter named ``ctx`` receives the full ``Context``. No manual context lookups needed.

5. **DAGResolver for graph merging**: Model operations are placed into the solver's loop structure automatically based on their ``depends_on`` and ``before`` declarations, then topologically sorted within each scope.

6. **Lazy initialization**: The BUILD stage produces ``InitStep`` descriptions, not live objects. This enables dependency validation, cycle detection, and deterministic ordering before any memory is allocated.
