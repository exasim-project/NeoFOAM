Project layout
==============

Where to find things in the ``neofoam`` package (``src/neofoam/``). Everything
is **config-driven**: a case is a set of validated pydantic configs that
serialize to OpenFOAM/JSON/YAML files. Each entry below is a one-line map; follow
the linked reference pages for the full API.

``io/``
    Binds config classes to files (``BaseConfig`` + ``@IOStrategy``), writes and
    deep-merges case files, and exposes the frontend-agnostic config-introspection
    surface (``schema.py``).

``fields/``
    Boundary-condition types, ``FieldValue`` scalar/vector values, and the
    synthesised ``0/<name>`` field schemas.

``foam/``
    ``fvSchemes``/``fvSolution`` config builders and FoamFile helpers.

``framework/``
    The Spec/Runtime core — see :doc:`/reference/framework/index`.

    - ``model/`` — ``ModelSpec``/``Model``, the decorator-driven model definition
      and its runtime. See :doc:`/reference/model/index`.
    - ``solver/`` — ``SolverSpec``/``Solver``, core/optional model selection and
      the case-free config schema. See :doc:`/reference/solver/index`.
    - ``initialization/`` — staged init (load → resolve → build) via ``InitStep``
      and ``InitializerBuilder``. See :doc:`/reference/initialization/index`.
    - ``graph/`` — operation-graph sort, resolution, validation, and
      visualization. See :doc:`/reference/graph/index`.
    - ``validation/`` — case-correctness checks: a registry of isolated checks
      with a structurally honest ``ok``.

``algorithms/solution_loop/``
    The pure-Python time loop, with an injectable constraint list and a named
    measurement registry; time-step rules are opt-in models.

``solver/incompressibleFluid/``
    The reference solver: its ``SolverSpec``, field wiring, and the
    PIMPLE/viscosity/turbulence/boussinesq/adaptiveTimeStep models.

``agent/``
    LLM case scaffolding: the pydantic-ai case-fill agent, forms, and the
    packaged marimo wizard template.

``core/``
    ``PluginSystem`` — discriminated registries for constraints, time-integration
    regimes, and model families.

``tooling/``
    The "above the library" layer: the stdlib-only ``Workspace`` path sandbox and
    other frontend/trust-boundary concerns.
