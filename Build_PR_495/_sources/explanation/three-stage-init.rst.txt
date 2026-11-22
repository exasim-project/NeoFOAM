Three-stage initialization
==========================

The three-stage init pipeline exists to **populate the Context that
the core abstraction reads from**. Operations described in
:doc:`operations-and-the-dag` cannot run until ``ctx["fields.U"]``,
``ctx["models.turbulence"]``, and the rest are in place; they also
cannot run until cross-model dependencies (for example Boussinesq
flipping ``use_boussinesq=True`` on the pressure-velocity algorithm)
are wired. LOAD / RESOLVE / BUILD is the pipeline that delivers both,
in the right order, before the first operation fires.

.. seealso::
   For the lifecycle overview that places this page in context, see
   :doc:`solver-structure`. For what happens *after* BUILD finishes, see
   :doc:`operations-and-the-dag`.

When a NeoFOAM solver starts, initialization is split into three
explicit stages plus an implicit verification step:

1. **LOAD** — read configuration files, detect active models,
   instantiate ``ModelRuntime`` objects. No interaction between models.
   No mesh, no field allocation.
2. **RESOLVE** — wire inter-model dependencies through
   ``ConfigContext``; models adapt their configuration to what other
   models exist.
3. **VERIFY** *(implicit)* — check that every active operation's
   ``@fvSchemes.add`` / ``@fvSolution.add`` requirements are satisfied
   by the loaded ``system/fvSchemes`` / ``system/fvSolution``. Raises
   before any allocation.
4. **BUILD** — produce ``InitStep`` objects that describe how to create
   runtime objects (fields, operators, models). The framework
   topologically sorts them and executes them in dependency order,
   producing the ``Context``.

.. mermaid::

   sequenceDiagram
       autonumber
       participant Caller
       participant Init as StagedInit
       participant Files as Case files<br/>(system/, constant/, 0/)
       participant Models as ModelRuntime(s)
       participant Cfg as ConfigContext
       participant Verify as Verifier
       participant Ctx as Context

       rect rgb(227, 242, 253)
           Note over Caller,Models: LOAD — read files, instantiate models in isolation
           Caller->>Init: run()  /  validate()
           Init->>Files: read fvSchemes, fvSolution,<br/>turbulenceProperties, …
           Files-->>Init: raw config dicts
           Init->>Models: instantiate per active model
           Models-->>Init: loaded configs<br/>(no cross-model interaction)
       end

       rect rgb(227, 242, 253)
           Note over Init,Cfg: RESOLVE — models adapt configs to peers
           Init->>Cfg: register loaded models by name
           Init->>Models: @resolve(config_context)
           Models->>Cfg: read peer configs,<br/>mutate own (e.g. use_boussinesq=True)
       end

       rect rgb(255, 243, 224)
           Note over Init,Verify: VERIFY (implicit) — fail fast before BUILD
           Init->>Verify: collect @fvSchemes.add /<br/>@fvSolution.add requirements
           Verify->>Files: cross-check against<br/>system/fvSchemes, system/fvSolution
           Verify-->>Init: list[VerificationError]
           alt validate() only
               Init-->>Caller: errors (skip BUILD)
           end
       end

       rect rgb(232, 245, 233)
           Note over Init,Ctx: BUILD — produce InitSteps and execute them
           Init->>Models: @build → list[InitStep]
           Init->>Init: topological_sort(init_steps)
           Init->>Ctx: execute steps<br/>(allocate mesh, fields, operators)
           Ctx-->>Caller: Context (solver-ready)
       end

What makes this hard
--------------------

OpenFOAM's solver ``main()`` reads the dictionary, allocates the mesh,
and constructs models in one pass — validation means running the
solver. That works when the only consumers are "the solver" and "the
case author who is about to run the solver." Three forces push against
the single-pass design once the surface goes Python:

- **Cross-model dependencies are order-sensitive.** The Boussinesq
  model needs to flip ``use_boussinesq=True`` on whichever
  pressure-velocity algorithm was selected — but only *after* the
  algorithm is loaded and *before* either of them allocates fields.
  With a single init step, the ordering rules end up encoded inside
  whichever model happens to run last, and adding a new model means
  revisiting that ordering.
- **Field allocation cycles.** ``p_rgh`` (Boussinesq) depends on
  ``mesh`` and ``p`` and ``rhok``; ``rhok`` depends on ``T``; ``T``
  depends on ``mesh``. The dependency graph is fine, but it has to be
  *expressed* somewhere, and "construct everything in the right
  order" buries that expression inside imperative code.
- **Validation that needs the resolved config but not the allocated
  fields.** A user wants to know "does my case satisfy the solver's
  requirements?" without paying the cost of mesh I/O. That implies
  LOAD + RESOLVE need to be runnable independently of BUILD — exactly
  what OpenFOAM's "validate by running" pattern cannot offer.

Mechanism: what each stage cannot do
------------------------------------

The constraints below are easy to break by accident, and the framework
relies on them holding:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Stage
     - Forbidden
     - Why it matters
   * - **LOAD**
     - Depending on other models' loaded state
     - Each ``@load`` runs in isolation; the framework provides no
       guaranteed ordering between models. Cross-model wiring belongs
       in RESOLVE.
   * - **RESOLVE**
     - Allocating fields or touching the mesh
     - No mesh exists yet, no ``InitStep`` has run. RESOLVE is
       config-only — it edits ``ConfigContext`` and returns.
   * - **BUILD**
     - Reading configuration files
     - It receives the resolved config from RESOLVE; reading files at
       this point bypasses verification and breaks the
       schema-introspection contract.

Hooked into ``incompressibleFluid``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The three stages are not framework abstractions floating in the
abstract — they are exactly the three decorators in
``src/neofoam/solver/incompressibleFluid/create_fields.py``:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Decorator in ``create_fields.py``
     - Stage
     - What ``incompressibleFluid`` does
   * - ``@init.load``
     - LOAD
     - ``PressureVelocityAlgorithm.detect_and_create()`` reads
       ``system/fvSolution`` to pick PIMPLE / SIMPLE / PISO;
       ``incompressibleFluidModel.detect_models()`` walks the plugin
       registry; ``FvSchemesConfig`` / ``FvSolutionConfig`` are loaded
       for verification. Returns a ``LoadResult``.
   * - ``@init.resolve``
     - RESOLVE
     - Each loaded model's ``run_resolve(config_context)`` is called.
       This is where Boussinesq flips ``use_boussinesq=True`` on the
       pressure-velocity algorithm, for example.
   * - ``@init.build``
     - BUILD
     - ``InitializerBuilder`` produces ``InitStep``\ s for the time/
       mesh, the pressure-velocity algorithm, ``laminarTransport``,
       ``turbulence``, and the optional models. The framework
       topologically sorts the steps and runs them, populating
       ``Context.fields`` and ``Context.models``.

Reading those three decorators side-by-side is the easiest way to see
the lifecycle on real code; everything else on this page is the
rationale for the split.

Trade-offs: why three, not two or four
--------------------------------------

**Why not two (LOAD + BUILD)?**
Without RESOLVE, every cross-model dependency has to fit inside one of
the two stages. LOAD cannot contain it (a model cannot see configs
that other models have not loaded yet); BUILD cannot contain it
cleanly (by then field allocations are already happening). Splitting
LOAD and BUILD with a RESOLVE step in the middle is what gives
"configs influence other configs" a place to live.

**Why not four (separate VERIFY)?**
Verification is a check, not an action — it does not produce state, it
just rejects bad input. Hiding it as an automatic step between RESOLVE
and BUILD means the user-facing surface stays at three stages while
the framework still gets to fail fast.

The honest cost is that solver authors and model authors have to know
which stage their code lives in, and respect the constraints. A single
constructor is harder to misuse and harder to extend; three stages is
easier to extend and easier to misuse. The framework's lint and tests
are what keep the constraints honest.

When this matters in practice
-----------------------------

With the constraints above in place, three user-visible capabilities
become cheap:

- ``StagedInit.validate()`` runs LOAD + RESOLVE + VERIFY without
  BUILD. Case validation costs file I/O plus config wiring — no mesh,
  no fields.
- ``StagedInit.solver_inputs()`` introspects every config class
  without running anything. See :doc:`schema-introspection`.
- ``StagedInit.scheme_inputs()`` introspects every required scheme
  without parsing a case.

In a single-stage init, each of those would require parsing the entire
pipeline. In a two-stage init, only the first would be possible.
