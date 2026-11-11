Schema introspection
====================

Three earlier design choices — :doc:`three-stage-init`,
:doc:`discriminated-unions`, and per-operation
``@fvSchemes.add`` / ``@fvSolution.add`` requirements — combine to make
case validation, schema export, and scheme introspection cheap.
This page traces that causal chain and then lists the user-visible
capabilities that fall out of it.

The three pieces (causal chain)
-------------------------------

The capability is built on three pieces, each of which exists for an
independent reason but combines to make introspection trivial:

**1. The three-stage init split makes validation cheap.**
``StagedInit.validate()`` walks the same code path the real solver
does, but stops before BUILD. LOAD and RESOLVE do not touch the mesh
and do not allocate fields — that is the entire point of the
three-stage pipeline (see :doc:`three-stage-init`). A typo in
``constant/turbulenceProperties`` is caught after LOAD without paying
for mesh I/O.

**2. The discriminated-union plugin registry exposes the schema.**
Every model's config is a Pydantic ``BaseConfig`` subclass; the
plugin registry rebuilds its discriminated union every time a new
plugin registers (see :doc:`discriminated-unions`). Calling
``model_json_schema()`` on the rebuilt union returns a JSON schema
that reflects the *currently-installed* plugin set — not whatever set
the framework shipped with.

**3. Per-operation requirements live on the operations themselves.**
Each operation declares its scheme/solution requirements with
``@fvSchemes.add(...)`` and ``@fvSolution.add(...)``.
``collect_requirements_from_models()`` aggregates them across every
active model; ``verify_fvschemes()`` and ``verify_fvsolution()``
check the loaded case against the aggregated set.

None of the three pieces was designed *for* schema introspection.
Each came from a different goal — multi-physics resolution, third-party
extensibility, declarative scheme contracts. Schema introspection is
what falls out when you have all three.

The capabilities that fall out
------------------------------

- ``StagedInit.validate()`` — run LOAD + RESOLVE + VERIFY against a
  case directory and return any ``VerificationError`` objects. No
  fields are allocated, no mesh is loaded.
- ``StagedInit.solver_inputs()`` — return a ``dict[str, type]`` of
  Pydantic config classes for every model the solver knows about.
  No case directory required.
- ``StagedInit.scheme_inputs()`` — return a Pydantic model whose
  fields are typed with the correct scheme unions (``DdtScheme``,
  ``DivScheme``, …) for every ``@fvSchemes.add`` requirement
  declared by the active operations.

What makes this hard (the alternative)
--------------------------------------

OpenFOAM validates by running. A misconfigured case fails at the first
solver step that touches the bad entry — sometimes minutes in, after
mesh decomposition has already consumed memory and walltime. The error
location (a stack trace inside ``RASModel::correct``) is rarely the
same as the configuration location (a typo in ``RAS.RASModel`` in
``constant/turbulenceProperties``). For the case author, the loop is
"edit, run, wait, read traceback, guess, edit again."

Three other consumers want something cheaper than that:

- **CI gates.** A test that says "every example case in the repo
  passes ``solver_validate``" needs to run in seconds across hundreds
  of cases. Allocating a mesh per case is too expensive.
- **UI form generation.** A web form that lets a user edit a case
  needs to know which fields are valid and what their types are
  *before* the user has filled in any values.
- **AI-assisted templating.** An LLM that drafts ``constant/`` and
  ``system/`` files for a target solver needs the schema, not the
  solver source code.

A "validate by running" pattern serves none of those well.

Trade-offs
----------

The capabilities have honest limitations:

- **Validation is only as good as the requirement declarations.** An
  operation that reads ``system/fvSolution`` directly (instead of
  declaring an ``@fvSolution.add(...)``) bypasses verification. The
  framework cannot detect the bypass; the linter and code review do.
- **Schema introspection holds at the model-config level, not at the
  resolved-runtime-value level.** ``solver_inputs()`` tells you that
  the Boussinesq model's ``thermalExpansion`` field is a ``float``;
  it does not tell you whether the value chosen for *your* case is
  numerically reasonable.
- **Plugin registry rebuild cost.** ``solver_inputs()`` is cheap
  *per call*, but the registry rebuild it depends on happens at
  every plugin import. For a process that imports many plugins, that
  cost is paid once at startup.
- **Validation is point-in-time.** A case that validated yesterday
  may not validate today if a plugin was upgraded and now requires a
  new scheme. That is the right behaviour, but it does mean
  ``validate() == 0 errors`` is not durable across plugin upgrades.

When this matters in practice
-----------------------------

The user-visible recipes are:

- :doc:`/auto_tutorials/example_04_configure_with_io` — use the schema
  export to drive config generation rather than hand-writing OpenFOAM
  dictionaries.

This capability is what delivers the *simplify usability* goal from
:doc:`goals-and-core-concepts` — typed configs, IDE autocomplete,
validate-before-solve, JSON schema for external tooling. It is not a
feature that was built; it is what falls out when the three earlier
design choices are paid for.
