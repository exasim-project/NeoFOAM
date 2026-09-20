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
    ``fvSchemes``/``fvSolution`` config builders, the typed ``fvSolution``
    algorithm blocks (``PIMPLE``/``PISO``/``SIMPLE``), and FoamFile helpers.

``framework/``
    The Spec/Runtime core — see :mod:`neofoam.framework`.

    - ``model/`` — ``ModelSpec``/``Model``, the decorator-driven model definition
      and its runtime. See :mod:`neofoam.framework.model`.
    - ``solver/`` — ``SolverSpec``/``Solver``, core/optional model selection and
      the case-free config schema. See :mod:`neofoam.framework.solver`.
    - ``initialization/`` — staged init (load → resolve → build) via ``InitStep``
      and ``InitializerBuilder``. See :mod:`neofoam.framework.initialization`.
    - ``graph/`` — operation-graph sort, resolution, validation, and
      visualization. See :mod:`neofoam.framework.graph`.
    - ``validation/`` — case-correctness checks: a registry of isolated checks
      with a structurally honest ``ok``.

``algorithms/solution_loop/``
    The pure-Python time loop, with an injectable constraint list and a named
    measurement registry; time-step rules are opt-in models.

``solver/incompressibleFluid/``
    The reference solver: its ``SolverSpec``, field wiring, and the
    PIMPLE/viscosity/turbulence/boussinesq/adaptiveTimeStep models.

``turbulence/``
    The momentum-transport plugin family: the bundled models (``models/``) and
    ``select_turbulence_model``, which builds the native-NeoN or the pybFoam
    fallback handle.

``viscosity/``
    The ``viscosityModel`` plugin family (native ``newtonian``), with a fallback to
    pybFoam's ``singlePhaseTransportModel``.

``tools/``
    Solver-agnostic preprocessing tools (``blockMesh``, ``snappyHexMesh``,
    ``checkMesh``), their registry, and the ``system/preprocess.yaml`` runner behind
    ``neofoam preprocess``.

``telemetry/``
    Opt-in OpenTelemetry performance tracing (``telemetry`` extra) and the trace
    reports behind ``neofoam telemetry``.

``cli/``
    The Typer app behind the ``neofoam`` command — see :doc:`cli`.

``agent/``
    LLM case scaffolding: the pydantic-ai case-fill agent and the form wiring
    shared with the case wizard.

``mcp/``
    The frontend-neutral tool layer: case-free ``f(solver, ...)`` functions returning
    pydantic DTOs (``tools.py``, ``dto.py``), the solver registry, and the FastMCP
    server behind ``neofoam mcp serve``. The case wizard calls the same functions.

``ui/``
    The case-wizard web UI behind ``neofoam ui`` (trame + JSONForms) — see
    :doc:`ui-architecture`.

    - ``app.py`` — the orchestrator: state seeding, controllers, layout.
    - ``forms.py``, ``form_schema.py``, ``boundary_forms.py`` — the form registry and
      the pydantic JSON Schema → JSONForms transform.
    - ``steps.py``, ``case_spec.py``, ``case_load.py``, ``review.py``, ``scaffold.py``
      — steps and model selection, form state ↔ case spec, reopening a case,
      findings, ``Allrun``/``Allclean``.
    - ``geometry*.py``, ``agent_panel.py``, ``sweep_*.py`` — the Geometry step, the AI
      chat drawer and the Parameters (sweep) step.
    - ``plugins/`` — the step-plugin interface (``neofoam.ui.steps`` entry points).
    - ``jsonforms_module/`` — the JS renderers and their checked-in bundle; see the
      ``README.md`` there.

``core/``
    ``PluginSystem`` — discriminated registries for constraints, time-integration
    regimes, and model families.

``tooling/``
    The "above the library" layer: the stdlib-only ``Workspace`` path sandbox and
    other frontend/trust-boundary concerns.
