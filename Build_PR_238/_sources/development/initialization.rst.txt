3-Stage Solver Initialization
=============================

Motivation
----------

The operation concept makes it possible to reorder, add or remove solver operations just by adding additional models.
However, this flexibility requires a more sophisticated initialization process to ensure that all models and their dependencies are properly configured before runtime objects (fields, meshes) are created.
The **3-stage initialization process** handles complex dependencies between solvers, models, and physics algorithms.


This ensures that configuration is loaded, dependencies are resolved, and runtime objects (fields, meshes) are built in the correct order.
To reduce the size of the solver code, the initialization logic is encapsulated in the ``StagedInit`` class and associated decorators that define each state in a separated file.

The Process: Load → Resolve → Build
------------------------------------

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
    - Only the InitSteps are defined here, but they are not executed until the Manager topologically sorts them.
    - A graph of dependencies is used for **Initialization** to ensure fields are created in dependency order (Mesh → U → Turbulence).

Usage Example
-------------

Define each stage with decorators on a ``StagedInit`` instance. The ``@load`` handler returns a ``LoadResult``,
the optional ``@resolve`` handler wires models via ``ConfigContext``, and the ``@build`` handler returns
a list of ``InitStep`` objects. Calling ``run()`` executes all three stages and returns a ``Context``.

.. code-block:: python

    from neofoam.framework.initialization.staged_init import StagedInit, LoadResult
    from neofoam.framework.initialization.helpers import field, init, model

    staged = StagedInit("IncompressibleSolver")

    # --- Stage 1: Load configuration and lightweight model objects ---
    @staged.load
    def load_config():
        transport = TransportModel()       # no fields yet
        turbulence = TurbulenceModel()
        return LoadResult(core_models=[transport], optional_models=[turbulence])

    # --- Stage 2 (optional): Wire cross-model dependencies ---
    @staged.resolve
    def resolve_dependencies(cfg):
        turbulence = cfg.get("turbulence")
        turbulence.transport_ref = cfg.get("transport")

    # --- Stage 3: Return InitStep recipes (the framework orders them) ---
    @staged.build
    def build_runtime(core_models, optional_models):
        return [
            # Root objects (no dependencies)
            init("mesh", create=lambda _ctx: create_mesh()),

            # Fields depend on mesh
            field("p", depends_on=["mesh"], create=lambda ctx: read_pressure(ctx["mesh"])),
            field("U", depends_on=["mesh"], create=lambda ctx: read_velocity(ctx["mesh"])),

            # Models depend on fields
            model("algo", create=lambda ctx: create_algorithm(ctx["fields.U"], ctx["fields.p"])),
        ]

    # --- Execute all 3 stages ---
    ctx = staged.run()

    # ctx is a Context object:
    #   ctx.mesh    == result of init("mesh", ...)
    #   ctx.fields  == {"U": ..., "p": ...}
    #   ctx.models  == {"algo": ...}

The framework topologically sorts the ``InitStep`` list, executes each initializer in order,
and populates the ``Context``. Circular or missing dependencies raise immediately.

Helper Functions
~~~~~~~~~~~~~~~~

Helper functions create ``InitStep`` objects with automatic naming conventions:

- ``init(name, create, depends_on)`` — general-purpose, no prefix (for mesh, runtime, algorithm, etc.)
- ``field(name, create, depends_on)`` — prefixes with ``fields.`` (e.g., ``field("U", ...)`` → name ``fields.U``)
- ``model(name, create, depends_on)`` — prefixes with ``models.``

Each initializer receives a ``context`` dict containing all previously created objects, keyed by their full name.
