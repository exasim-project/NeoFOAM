Solver structure
================

A NeoFOAM solver is **a sequence of custom operations that act on a
shared Context**. The solver author declares a standard order of
operations through the ``execution_graph``; plugin models contribute
additional operations that modify that order. The framework merges
them and runs the result.

This page is the conceptual overview. The detail behind each piece —
the data model, the merger algorithm, the decorators — lives in
:doc:`operations-and-the-dag`.

Operations change the Context
-----------------------------

The ``Context`` is a small data class that holds the state shared by
all operations. It is the only coupling mechanism between them —
there is no other shared state.

.. code-block:: python

    # src/neofoam/framework/context.py
    class Context(BaseModel):
        fields: dict[str, Any]   # e.g. fields["U"], fields["p"], fields["phi"]
        models: dict[str, Any]   # e.g. models["turbulence"], models["pressure_velocity"]
        mesh: Any = None
        runtime: Any = None      # OpenFOAM runtime (time, mesh handle)

Every operation either takes the Context directly or asks for a
specific entry from it. Two examples from
``src/neofoam/solver/incompressibleFluid/incompressibleFluid.py``:

.. code-block:: python

    # from incompressibleFluid.py
    @incompressibleFluid.operation()
    def increment_time(self: Any, ctx: Context) -> None:
        Info(f"Time = {ctx.time.timeName()}")
        ctx.time.increment()

    # from models/pressure_velocity/pimpleAlgorithm.py
    @pimple.operation(operation_number="2.1")
    @fvSchemes.add(ddt="ddt(U)", div="div(phi,U)", grad="grad(U)",
                   laplacian="laplacian(nuEff,U)")
    @fvSolution.add("U")
    def momentum(
        U: volVectorField,
        phi: surfaceScalarField,
        p: volScalarField,
        turbulence: Annotated[TurbulenceModel, "models"],
        pimple_control: Annotated[PimpleControl, "models"],
    ) -> FieldUpdates:
        UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U)
                              + turbulence.divDevReff(U))
        UEqn.relax()
        if pimple_control.momentumPredictor():
            pyf.solve(UEqn + fvc.grad(p))
        return FieldUpdates({"UEqn": UEqn, "U": U})

Two ways of touching the Context show up here:

- **Direct read** — ``increment_time`` takes ``ctx: Context`` and
  reaches into ``ctx.time`` to ask the OpenFOAM runtime for the
  current time and to advance it.
- **Parameter injection** — ``momentum`` doesn't take ``ctx`` at all.
  The framework inspects the function signature and fills each
  parameter from the Context by name: bare typed parameters
  (``U``, ``phi``, ``p``) come from ``ctx.fields``; parameters
  tagged ``Annotated[..., "models"]`` (``turbulence``,
  ``pimple_control``) come from ``ctx.models``. If the name is not
  there, the parameter arrives as ``None``. The full rules live in
  :doc:`parameter-injection`.

Operations that mutate fields return a ``FieldUpdates`` dict. The
framework writes the entries back into ``ctx.fields`` after the
operation returns — so the ``momentum`` operation above publishes the
updated velocity ``U`` and the momentum-equation matrix ``UEqn``
under those names in the Context, where the next operation
(``continuity``) will pick them up.

The decorators
~~~~~~~~~~~~~~

Three decorators wrap the operations in the snippet above. Each one
solves a different *"how does this function plug into the framework?"*
problem:

- ``@solver.operation(...)`` / ``@model.operation(...)`` — registers
  the function as a named operation on the solver or the plugin
  model. The ``execution_graph`` later looks it up by name
  (e.g. ``ops["increment_time"]``); without this decorator the
  function is just a function, invisible to the framework. Optional
  keyword arguments (``depends_on``, ``before``,
  ``operation_number``) tell the merger where the operation should
  land relative to others (used in section 3).
- ``@fvSchemes.add(...)`` — declares which finite-volume scheme
  entries the operation will use from ``system/fvSchemes`` (the
  ``ddt``, ``div``, ``grad``, ``laplacian`` terms it assembles).
  The framework aggregates these declarations across every active
  operation and verifies the case's ``fvSchemes`` file *before* the
  solver runs. A missing entry is caught at startup, not mid-loop.
- ``@fvSolution.add("U")`` — declares which fields the operation
  solves for, so the framework can check that ``system/fvSolution``
  carries the matching linear-solver settings. Same validation
  story as ``fvSchemes``.

Read together, these decorators are how an operation announces
*what it needs from the case* — placement metadata for the merger,
scheme requirements, solver settings. The validation that falls out
of these declarations is covered in :doc:`schema-introspection`.

For the full Context API, the parameter-injection contract, and the
``FieldUpdates`` write-back pattern, see :doc:`operations-and-the-dag`.

The execution graph defines the standard operations
---------------------------------------------------

The solver declares its loop topology in
``@solver.execution_graph_step``. This is the solver's *standard*
ordering — what the solver does when no plugin models are loaded.
Reading the ``execution_graph`` of ``incompressibleFluid`` is the
easiest way to see what the solver does. From
``src/neofoam/solver/incompressibleFluid/incompressibleFluid.py``:

.. code-block:: python

    @incompressibleFluid.execution_graph_step
    def execution_graph(self, ...):
        ops = self.operations
        builder = StepBuilder()

        # Pressure-velocity algorithm (pimple/simple/piso) chosen at LOAD
        algorithm_model = self.state.core_models[0]
        algo_ops = Operations(
            algorithm_model._build_operations_for(algorithm_model)
        )

        time_loop_op = Operation(
            func=IterativeOp(TimeLoop()),
            metadata=OperationMetadata(op_name="time_loop"),
        )

        with builder.loop(time_loop_op) as time_builder:
            time_builder.step(ops["set_time_step"])
            time_builder.step(ops["increment_time"])

            with time_builder.loop(algo_ops["inner_loop"]) as inner:
                inner.step(algo_ops["momentum"])
                inner.step(algo_ops["continuity"])
                inner.step(ops["turbulence_correction"])

            time_builder.step(ops["write_output"])

        # Collect ops contributed by optional plugin models
        model_ops = Operations()
        for model in self.state.optional_models:
            model_ops.add(model.operations)

        return builder, model_ops

The nested ``with builder.loop(...)`` blocks read top-to-bottom as
the solver's standard graph:

.. mermaid::

   flowchart TB
       subgraph TIME ["time_loop"]
           direction TB
           ST["set_time_step"]
           IT["increment_time"]
           subgraph INNER ["inner_loop (algorithm)"]
               direction TB
               M["momentum"]
               C["continuity"]
               T["turbulence_correction"]
               M --> C --> T
           end
           W["write_output"]
           ST --> IT --> INNER --> W
       end

       style TIME fill:#E3F2FD
       style INNER fill:#BBDEFB

``self.state`` exposes the models resolved at LOAD time:
``core_models`` (mandatory variants like the pressure-velocity
algorithm — exactly one was picked from ``pimple`` / ``simple`` /
``piso``) and ``optional_models`` (plugin contributions discovered at
LOAD; see :doc:`discovery-mechanisms`).

``execution_graph`` returns *two* things — the structural ``builder``
above and ``model_ops`` (the contributions gathered from plugin
models). The framework merges them, which is what the next section
is about.

For the decorator API (``@solver.operation``,
``@solver.execution_graph_step``, ``@solver.initializer``) and the
full walkthrough, see :doc:`operations-and-the-dag`.

Additional models modify the operations
---------------------------------------

Plugin models contribute their own operations and declare where those
should land via ``depends_on`` / ``before`` / ``operation_number``
constraints. The DAG resolver merges the contributions into the
solver's standard ordering, producing the actual run-time graph.

The two plugins below ship with the framework. Boussinesq inserts
*two* operations inside the algorithm's inner loop, between
``momentum`` and ``continuity``. From
``src/neofoam/solver/incompressibleFluid/models/boussinesq.py``:

.. code-block:: python

    boussinesq = Model("boussinesq").register_with(incompressibleFluidModel)

    @boussinesq.operation(operation_number="2.5", depends_on=["momentum"])
    @fvSchemes.add(ddt="default", div="div(phi,T)", laplacian="default")
    @fvSolution.add("T")
    def solve_energy(self, T, phi, turbulence, alphat):
        ...
        return FieldUpdates({"T": T, "alphat": alphat})

    @boussinesq.operation(operation_number="2.7", depends_on=["solve_energy"])
    def update_rhok(self, T, rhok):
        ...
        return FieldUpdates({"rhok": rhok})

A passive-scalar transport plugin adds a single operation that runs
*after* ``continuity``. The plugin lives outside the solver source
tree (see :doc:`/auto_tutorials/example_02_passive_scalar_plugin`)
but the registration pattern is the same:

.. code-block:: python

    passive_scalar = Model("passive_scalar").register_with(incompressibleFluidModel)

    @passive_scalar.operation(depends_on=["continuity"])
    @fvSchemes.add(ddt="ddt(s)", div="div(phi,s)", laplacian="laplacian(D,s)")
    @fvSolution.add("s")
    def solve_s(self, s, phi):
        cfg: PassiveScalarConfig = self.config
        D = pyf.dimensionedScalar("D", pyf.dimViscosity, cfg.D)
        sEqn = fvScalarMatrix(fvm.ddt(s) + fvm.div(phi, s) - fvm.laplacian(D, s))
        sEqn.solve()
        return FieldUpdates({"s": s})

With both plugins active, the merger produces this graph:

.. mermaid::

   flowchart TB
       subgraph TIME ["time_loop (with boussinesq + passive_scalar)"]
           direction TB
           ST["set_time_step"]
           IT["increment_time"]
           subgraph INNER ["inner_loop"]
               direction TB
               M["momentum"]
               SE["solve_energy ★<br/>boussinesq<br/>op_number=2.5<br/>depends_on=momentum"]
               UR["update_rhok ★<br/>boussinesq<br/>op_number=2.7<br/>depends_on=solve_energy"]
               C["continuity"]
               SS["solve_s ★<br/>passive_scalar<br/>depends_on=continuity"]
               T["turbulence_correction"]
               M --> SE --> UR --> C --> SS --> T
           end
           W["write_output"]
           ST --> IT --> INNER --> W
       end

       style TIME fill:#E3F2FD
       style INNER fill:#BBDEFB
       style SE fill:#FCE4EC
       style UR fill:#FCE4EC
       style SS fill:#E0F7FA

★ = inserted by a plugin model. The placement comes entirely from
the operation metadata on each plugin's decorators — the
``incompressibleFluid`` source has no idea ``solve_energy``,
``update_rhok``, or ``solve_s`` exist.

This is the payoff of the core abstraction: a third party adds a
plugin model and the solver picks up the new operations at the
correct point in the loop, without source edits.

For the merger algorithm (topological sort, ``operation_number``
tie-breaks, scope inference) see :doc:`operations-and-the-dag`. For
how plugin models are discovered at LOAD time, see
:doc:`discovery-mechanisms`.

Where to go next
----------------

- :doc:`operations-and-the-dag` — data model and merger algorithm.
- :doc:`spec-runtime` and :doc:`three-stage-init` — how operations
  and Context are set up before the loop runs.
- :doc:`discriminated-unions` and :doc:`discovery-mechanisms` — how
  plugin models register and how the solver picks them up.
- :doc:`schema-introspection` — what the typed configs buy you.
