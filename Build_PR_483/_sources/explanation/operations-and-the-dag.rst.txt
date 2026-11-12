Operations and the DAG
======================

This page describes the **core abstraction** of the library: a NeoFOAM
solver is a sequence of custom operations that act on a shared
``Context``. Every other page in this section explains a piece of
infrastructure that exists to make this work — definition-time specs,
init-time Context population, plugin registries. This page is the one
that describes what the user is actually writing.

The solver itself only declares the operations *it* contributes. Models
contributed by the plugin registry or by core specs add their own
operations, and those operations need to land in the right loop scope,
in the right order. This page is the data model and the merger pattern
that make that work.

.. seealso::
   For how the Context gets populated *before* the operations run, see
   :doc:`three-stage-init`. For the lifecycle overview that places this
   page in context, see :doc:`solver-structure`.

What makes this hard
--------------------

OpenFOAM's solvers hardcode the order of solver steps in C++:
``UEqn.solve()`` → ``pEqn.solve()`` → ``correctBoundaryConditions()``,
each at a known line number. Adding a Boussinesq term means editing
the solver to call into the model at the correct point. The sequence
*is* the solver's source code.

That works as long as one author owns the solver and ships it as a
single binary. It breaks when:

- Optional models contributed by third-party packages need to land in
  a precise place in the loop (e.g. the Boussinesq energy equation
  has to run *before* continuity).
- Several optional models compete for the same loop scope and need a
  defined ordering between them (e.g. two source terms acting on
  momentum).
- The same solver should accept a different *set* of optional models
  per case, without recompilation.

NeoFOAM moves the ordering metadata onto the operations themselves.
The solver declares a tree of named operations and loop scopes;
models attach operations with ``depends_on``, ``before``, and
``operation_number`` constraints; a resolver merges them.

Mechanism
---------

The smallest unit: ``Operation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``Operation`` is a callable wrapped with metadata. Three kinds of
callable wrappers exist:

- ``SequentialOp(fn)`` — runs once when the operation executes.
- ``IterativeOp(fn)`` — repeatedly calls ``fn(ctx)``; while it returns
  ``True``, the operation's sub-operations run again. This is how
  time-loops and PIMPLE outer-corrector loops are expressed.
- ``ConditionalOp(fn)`` — gates execution on a predicate;
  sub-operations run only when the predicate returns ``True``.

The metadata (``OperationMetadata``) carries the operation's name, an
optional ``OperationNumber`` (used as a tie-breaker during topological
sort), declared ``depends_on`` and ``before`` constraints, and a
visualisation hint (shape, colour) for the DAG view.

The container: ``Operations``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``Operations`` instance is a list of operations with a ``run(ctx)``
method that executes them in order. ``Operation.run(ctx)`` dispatches
on the wrapper kind: sequential operations run once, iterative
operations loop their sub-operations, and conditionals gate them.

The structure is flat-or-nested: top-level operations may contain
``sub_operations`` (themselves ``Operation`` instances). Time loops and
algorithm-correction loops are nested operations whose sub-operations
are the per-iteration body.

The composer: ``StepBuilder``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``StepBuilder`` is the fluent interface used to *build* an
``Operations`` tree::

    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(set_time_step_op)

        with time_builder.loop(inner_loop_op) as inner:
            inner.step(momentum_op)
            inner.step(continuity_op)

        time_builder.step(write_output_op)

It exposes two methods, ``step()`` and ``loop()``. ``loop()`` returns a
new builder scoped to the new loop's body and is also a context
manager, so the nested-``with`` pattern reads as cleanly as imperative
construction. The output is an ``Operations`` tree describing the
solver's loop topology.

The shared bus: ``Context``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operations don't talk to each other directly — they read from and write
to a shared ``Context`` object. The Context is what BUILD produces and
what the runtime carries from one operation to the next:

.. code-block:: python

    class Context(BaseModel):
        fields: dict[str, Any]   # e.g. fields["U"], fields["p"], fields["phi"]
        models: dict[str, Any]   # e.g. models["turbulence"], models["pressure_velocity"]
        mesh: Any = None
        runtime: Any = None      # time/mesh handle from pybFoam

Operation bodies see the Context two ways:

- **Direct access.** ``ctx.time.timeName()``,
  ``ctx.time.write(True)``, ``ctx["fields.U"]``,
  ``ctx["models.laminarTransport"]``. The string keys
  (``"fields.U"``) come from the BUILD-time
  ``InitStep`` that allocated each entry — that's why ``create_fields.py``
  declares ``depends_on=["fields.U", "fields.phi"]`` on dependent
  steps.
- **Field updates.** Operations that mutate fields return a
  ``FieldUpdates`` (a dict of names → new values); the framework
  applies the updates back to ``ctx.fields`` after the operation
  returns. ``turbulence_correction`` in ``incompressibleFluid.py``
  shows the pattern: it returns ``FieldUpdates({})`` even when it has
  nothing new to publish.

The Context is *the* coupling mechanism between operations. A model
contributing a new equation reads its inputs from ``ctx.fields`` /
``ctx.models``, runs its solve, and either mutates state in place
(through the C++ runtime) or publishes a ``FieldUpdates`` dict.

Parameter injection
~~~~~~~~~~~~~~~~~~~

Operation signatures don't have to take a single ``ctx`` — they can
declare typed parameters (fields and models, by name) that the
framework wires in before the call. This is *parameter injection*;
see :doc:`parameter-injection` for the rules, the resolver
behaviour, and the trade-offs. Two consequences worth pinning down
here, since the merger relies on them:

- **Operations don't fish in dictionaries.** The framework hands
  each operation exactly the parameters it asked for; missing
  optional models arrive as ``None`` so the operation body can
  guard with ``if model:``.
- **The "models" namespace is single-flat.** Both core models
  (pimple/simple/piso, CFL) and optional models (boussinesq,
  transport, turbulence) land in the same lookup table; the
  parameter name has to match the model's registered name. That's
  why a solver can declare
  ``pressure_velocity: Annotated[Optional[Any], "models"]`` and
  have it filled regardless of which algorithm variant was chosen.

The merger: ``DAGResolver``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A solver's structural builder describes only what the solver itself
contributes. Models contributed via the plugin registry and core specs
add their own operations — but they need to land in the correct loop
scope, and they need to respect declared ``depends_on`` constraints.
``DAGResolver.resolve(builder, additional_ops)`` does this in four
steps:

1. **Tag.** Walk the builder tree, tagging each operation with its
   enclosing scope ("root", "time_loop", "inner_loop", ...). Tag each
   model operation with the scope inferred from its dependency targets
   (innermost scope wins).
2. **Build a global graph.** Add every named non-loop operation as a
   node; add edges for each ``depends_on`` (dep → node) and ``before``
   (node → dep). Loop operations are *structural* — they are
   containers, not nodes in this graph.
3. **Sort.** Topologically sort with a custom key:
   ``(operation_number is None, operation_number, node_name)``.
   Operations with an ``operation_number`` come first in numeric
   order; unnumbered operations break ties alphabetically.
4. **Rebuild.** Produce a fresh ``StepBuilder`` whose nesting matches
   the original loop structure, but whose per-scope ordering reflects
   the resolved dependencies. Loop operations are re-injected after
   their last dependency in the parent scope.

.. mermaid::

   flowchart LR
       subgraph SOLVER ["Solver StepBuilder (input)"]
           direction TB
           S_TIME["time_loop"]
           S_TIME --> S_SET["set_time_step"]
           S_TIME --> S_INNER["inner_loop"]
           S_INNER --> S_MOM["momentum"]
           S_INNER --> S_CONT["continuity"]
           S_TIME --> S_WRITE["write_output"]
       end

       subgraph MODEL ["additional_ops (model contributions)"]
           direction TB
           M_ENERGY["energy_eqn<br/>(boussinesq)<br/>op_number=2.5<br/>depends_on=momentum"]
           M_TURB["turbulence<br/>(spalart_allmaras)<br/>depends_on=continuity"]
       end

       SOLVER --> RES{{"DAGResolver.resolve"}}
       MODEL --> RES

       subgraph MERGED ["Resolved StepBuilder (output)"]
           direction TB
           R_TIME["time_loop"]
           R_TIME --> R_SET["set_time_step"]
           R_TIME --> R_INNER["inner_loop"]
           R_INNER --> R_MOM["momentum"]
           R_INNER --> R_ENERGY["energy_eqn"]
           R_INNER --> R_CONT["continuity"]
           R_INNER --> R_TURB["turbulence"]
           R_TIME --> R_WRITE["write_output"]
       end

       RES --> MERGED

       style SOLVER fill:#E3F2FD
       style MODEL fill:#FFF3E0
       style MERGED fill:#E8F5E9
       style RES fill:#FFEBEE
       style M_ENERGY fill:#FFF3E0
       style M_TURB fill:#FFF3E0
       style R_ENERGY fill:#FFE0B2
       style R_TURB fill:#FFE0B2

The diagram shows two model operations being merged into the solver's
``inner_loop`` scope: the Boussinesq energy equation lands between
``momentum`` and ``continuity`` because of its ``depends_on=momentum``
plus ``operation_number=2.5``; the turbulence update lands after
``continuity`` because of its ``depends_on=continuity``. The solver
itself never named the model operations.

Trade-offs
----------

The merger pattern has two costs that the OpenFOAM hardcoded-loop
baseline does not pay.

**The resolver is non-trivial code.** The four-step merge operates on
a non-trivial data structure. Bugs in the resolver are hard to debug
because the failure mode — operations in the wrong order — is not
always immediately visible in the simulation result. The framework
ships a DAG visualiser to mitigate this, but the diagnostic is
"render the graph, look at it" rather than "read the solver source."

**Operation numbers are folklore.** ``operation_number`` is a
tie-breaker for operations that have no dependency relationship with
each other. The framework prefers ``depends_on`` strings (matching
another operation by name) because they survive renames and do not
require coordination across modules. Numbers are a last resort — but
they are also the only mechanism when two unrelated correction steps
both need to run somewhere inside a loop and the user wants a
deterministic order.

The Boussinesq model, for instance, picks ``"2.5"`` and ``"2.7"``
(see ``src/neofoam/solver/incompressibleFluid/models/boussinesq.py``)
to anchor its operations at specific phases of the inner loop. The
values are intentional — they sit between the canonical
``momentum`` (~2) and ``continuity`` (~3) steps — but a reader of
those numbers in isolation has no way to know that. Treat them as
folklore that should be paired with a comment naming the anchoring
operation; the resolver does the right thing, but the human reader
needs the context.

When this matters in practice
-----------------------------

Two tutorials exercise this directly:

- :doc:`/auto_tutorials/example_02_passive_scalar_plugin` adds a new
  scalar-transport equation as a model contribution. The model's
  operation lands in the inner loop without the solver knowing the
  model exists — the merger places it.
- :doc:`/auto_tutorials/example_03_build_a_solver` builds a complete
  ``StepBuilder`` tree from scratch and includes a worked example of
  an ``InitializationGraphError`` cycle. Seeing the DAG fail once
  before debugging it for real is part of the lesson.
