Goals & core concepts
=====================

This section explains the *why* behind NeoFOAM's design. Read these
pages to build a mental model of how
``src/neofoam/solver/incompressibleFluid/`` is wired. For step-by-step
recipes see :doc:`/how-to/install`; for guided lessons see the tutorials
listed in :doc:`/index`.

The core abstraction
--------------------

A NeoFOAM solver is **a sequence of custom operations that act on a
shared Context**. The solver author declares the loop topology and
contributes some operations; plugin models contribute more; the
framework merges them into one ordered DAG and runs them against a
``Context`` object that holds fields, models, the mesh, and the
runtime.

That is the whole library, in one sentence. Everything else exists to
make it work.

.. mermaid::

    flowchart LR
        OP1["Operation A<br/>(solver)"]
        OP2["Operation B<br/>(plugin)"]
        OP3["Operation C<br/>(solver)"]
        OP4["Operation D<br/>(plugin)"]
        CTX[("Context<br/>fields · models<br/>mesh · runtime")]

        OP1 -->|reads/writes| CTX
        CTX --> OP2
        OP2 -->|reads/writes| CTX
        CTX --> OP3
        OP3 -->|reads/writes| CTX
        CTX --> OP4
        OP4 -->|reads/writes| CTX

        style CTX fill:#FFF3E0
        style OP1 fill:#E3F2FD
        style OP2 fill:#FCE4EC
        style OP3 fill:#E3F2FD
        style OP4 fill:#FCE4EC

The four goals
--------------

#. **Multi-physics code** — run multiple coupled physics
   (turbulence + transport + optional models like Boussinesq) inside one
   solver, without solver-side edits per combination.
#. **Simplify usability** — typed configs, IDE autocomplete, validation
   before solve, JSON schema for external tooling.
#. **Extensibility** — third-party packages contribute physics models,
   schemes, and operations as plugins, without forking the solver.
#. **Integration of other numerical methods** — alternate numerical
   backends (NeoN GPU kernels, Ginkgo solvers) plug in behind a typed
   interface.

The core feature that delivers the goals
----------------------------------------

The **DAG-resolved operation graph** (see :doc:`operations-and-the-dag`)
is what carries every goal:

- Solver and plugin operations are merged and topologically sorted into
  one loop — that is *multi-physics code* without solver edits.
- The merger places third-party operations in the right scope based on
  declared ``depends_on`` / ``before`` constraints — that is
  *extensibility*.
- The Pydantic configs that operations depend on are typed, validated,
  and schema-exported — that is *simplify usability*.
- The compute called inside operations runs through ``pybFoam`` /
  NeoN — that is *integration of other numerical methods*.

Read :doc:`operations-and-the-dag` first if you want one page that
shows the core in action.
