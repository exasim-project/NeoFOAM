Spec and Runtime
================

This split is **enabling infrastructure for the core abstraction**
described in :doc:`operations-and-the-dag`. Operations are defined
once at module-import time (the *Spec*) and instantiated per case
(the *Runtime*) so that the same operation definition can back many
independent solver runs without state leaking through module globals.

.. seealso::
   For the lifecycle overview that places this page in context, see
   :doc:`solver-structure`.

NeoFOAM splits every model and solver into two layers:

- A **Spec** (``ModelSpec``, ``SolverSpec``) — the immutable definition.
  Created once at module-import time. Decorators like ``@spec.load``,
  ``@spec.build``, and ``@spec.operation(...)`` register callables onto
  the spec without executing anything.
- A **Runtime** (``ModelRuntime``, ``SolverRuntime``) — the mutable
  per-instance state, created from a spec via
  ``spec.instantiate(...)``. Holds the loaded config, resolved
  dependencies, and (for solvers) the argv and ``SolverState``.

The mental model is:

.. code-block:: text

    Module Import Time              Solver Startup
    ──────────────────              ──────────────
    ModelSpec   (definition)  ──►   ModelRuntime  (instance)
    SolverSpec  (definition)  ──►   SolverRuntime (instance)

What makes this hard
--------------------

OpenFOAM C++ leans hard on singleton-with-class-level-state for
turbulence models, transport models, fvOptions, and runtime-selection
factories. The class definition *is* the registry entry; the registry
entry *is* the configuration target; ``New(...)`` allocates and
configures in one step. That pattern works for one solver running once
in one process — but it breaks when:

- Tests want to spin up several independent runtimes of the same model
  in one process.
- Multi-domain simulations want to hold multiple runtimes of one spec
  concurrently (different regions, different boundary conditions).
- Schema-introspection and UI tooling want to ask "what could be
  loaded?" without committing to loading anything.

A Python-first surface where testability and concurrency matter cannot
inherit the singleton pattern unchanged.

Mechanism
---------

The split has three concrete payoffs:

**Multiple independent runs from one definition.**
A spec is essentially a class definition with extra structure — defined
once at import time and then re-used. Tests can spin up many
independent runtimes without leaking state through module globals.

**Plugin discovery without instantiation.**
The plugin registry stores ``ModelSpec`` instances, not runtimes. That
matters because schema introspection (``solver_inputs()``,
``scheme_inputs()``) and UI generation need to know *what could be
loaded* without committing to loading anything. If specs and runtimes
were the same object, plugin discovery would be a side-effecting
operation — incompatible with cheap ``--list-plugins`` calls or with
editor integrations that build forms from schemas.

**Explicit lifecycle.**
By the time a runtime exists, LOAD has already happened. By the time
operations execute, BUILD has already happened. The spec/runtime split
makes "what's allowed when" obvious in the API: you cannot accidentally
call a runtime-only method on a spec, and you cannot register a new
``@spec.operation`` after the runtime is built.

What lives where
~~~~~~~~~~~~~~~~

================================  =====================================================
On the **Spec**                   On the **Runtime**
================================  =====================================================
Decorator-registered callables    Loaded ``config`` (mutated by RESOLVE)
Plugin-system registration        Per-instance ``name`` (manifest entries override)
Class-level ``_config_class``     ``state.core_models``, ``state.optional_models`` (solver only)
Operation definitions             ``argv`` (solver only)
Detect / load / build hooks       Methods to *run* the registered hooks (``run_resolve``, ``run_build``, ``operations``)
================================  =====================================================

Trade-offs
----------

The cost is one indirection: writing a model takes two parts (the spec
declaration plus the runtime methods that the framework calls on it),
where a single class would feel more compact. The benefit is everything
above. For a framework whose top-priority capability is *runtime model
composition*, that is a price worth paying — but it is a price, and a
reader coming from OpenFOAM C++ will notice it.

When this matters in practice
-----------------------------

The split is what makes the following possible:

- **Schema-introspection without instantiation.** ``solver_inputs()``
  walks the registered specs and returns Pydantic config classes
  without running any model code. See :doc:`schema-introspection`.
- **Independent test runtimes.** Each test creates its own runtime,
  with its own loaded config — no module-global cleanup between tests.
- **Multi-domain simulations.** The fluid solver and the solid solver
  in a conjugate-heat-transfer setup may be runtimes of two different
  specs; nothing prevents both from being active simultaneously.
- **Late binding of name.** Manifest-loaded models override the
  runtime ``name`` so that the same spec can appear multiple times
  with different parameters (e.g. several heat sources).

The Spec/Runtime contract surfaces in :doc:`three-stage-init` (the
spec is created at import time, the runtime is created at LOAD) and in
:doc:`discriminated-unions` (the registry holds specs, not instances).
