Main Components
===============


As introduced in the architecture overview, the NeoFOAM solver framework tries to make the solver more modular and easier to extend by breaking down the solver into smaller components.



Main Concepts
^^^^^^^^^^^^^

The main concept of the NeoFOAM solver framework is to represent a CFD solver a series of operations that update fields based on governing equations.

.. hint::

   A CFD solver is a complex way to update fields with various operations based on governing equations.


A context objects holds all relevant data for the solver, including fields and models and is passed to each operation that modifies the context.
The context object is simply a simple data container that holds references to all relevant data structures:

.. code-block:: python

    class Context(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        fields: dict[str, Any]
        models: dict[str, Any]
        mesh: Any = None
        runtime: Any = None


Each operation is a class that implements a specific functionality, such as updating a field based on a governing equation or applying boundary conditions.
It can be created by using the ``@Solver.operation`` or ``@Model.operation`` decorator.
The decorator method flags the bounded method as an operation and collects metadata such as operation name, operation number, and dependencies.


.. code-block:: python

    @Solver.operation
    def solve_momentum(self, U, p) -> FieldUpdates:
        # ... computation
        return FieldUpdates({"U": U_new})

Additionally, decorated methods do not manually fetch data from the context. Instead, the framework inspects the method signature to determine which fields or models are required for the operation and automatically injects them when the method is called.
Consequently, the user only needs to declare the required inputs as method arguments, and the framework takes care of providing the correct data at runtime.
This keeps each operation independent of global state and makes it easier to test in isolation, since it can be executed by simply passing in the required arguments.
It also improves code readability and maintainability by making the dependencies of each operation explicit.
The dependencies between the operations can be visualized, which helps to understand the overall structure of the solver.


Operations
^^^^^^^^^^

There are three main types of operations in the framework:

* ``SequentialOp`` represents a regular operation in the solver sequence and contains a single function to execute.
* ``ConditionalOp`` represents an operation that is executed only if a certain condition is met, and has multiple sub-operations.
* ``IterativeOp`` represents an operation that is executed repeatedly while a certain condition is met, and has multiple sub-operations.

These operations types can be visualized as follows and are the building blocks of a solver workflow:

.. mermaid::

    flowchart TD

        subgraph ConditionalOp
            C1[Condition]
            CS2[SubOp 1]
            CS3[SubOp 2]
            C1 -- if True --> CS2
            CS2 --> CS3
        end

        subgraph IterativeOp
            I1[Loop Condition]
            IS2[SubOp 1]
            IS3[SubOp 2]
            I1 -- while True --> IS2
            IS2 --> IS3
            IS3 -- repeat --> I1
        end

        subgraph SequentialOp
            So1[Op 1]
            So2[Op 2]
            So3[Op 3]
            So1 --> So2
            So2 --> So3
        end

The idea is that complex solver workflows can be constructed by combining these basic operation types in a hierarchical manner.


All operations are stored as instances of the ``Operation`` class.
``Operation`` wraps a callable with a single ``metadata`` field of type ``OperationMetadata``;
properties on ``Operation`` (``operation_name``, ``operation_number``, ``depends_on``, ``before``, ``shape``, ``color``, ``domain_name``) delegate to that metadata.

.. code-block:: python

    @dataclass
    class Operation:
        """A concrete operation class that wraps a function with metadata."""

        func: Union[ConditionalOp, IterativeOp, SequentialOp]
        metadata: OperationMetadata = field(default_factory=OperationMetadata)
        level: int = 0
        sub_operations: list["Operation"] = field(default_factory=list)

To construct an ``Operation``, pass the metadata explicitly:

.. code-block:: python

    from neofoam.framework.operations import Operation, SequentialOp
    from neofoam.framework.types import OperationMetadata, OperationNumber

    op = Operation(
        func=SequentialOp(my_func),
        metadata=OperationMetadata(
            op_name="solve_momentum",
            operation_number=OperationNumber("1.0"),
            depends_on=["set_time_step"],
        ),
    )

The solver framework gathers all ``Operation`` instances defined in the solver and model classes and constructs a workflow that can be executed in sequence.
The resulting workflow is represented by the ``Operations`` class that is a container for all operations in the solver:

.. code-block:: python

    class Operations:
        def __init__(self, operations: list[Operation] | None = None) -> None:
            self.ops = operations if operations is not None else []

        def run(self, ctx: Context) -> None:
            self.print_tree()
            for operation in self.ops:
                operation.run(ctx)

It provides a ``run`` method that executes all operations in sequence, passing the context object to each operation.

.. note::
   ``Operations.run`` always calls ``self.print_tree()`` before executing — every call writes the operation tree to stdout.
   In normal solver usage ``run`` is invoked once on the resolved top-level container, so this only prints at startup.


Conditions
^^^^^^^^^^

Conditions are used to control the execution flow of operations in the solver framework and are used in ``ConditionalOp`` and ``IterativeOp``.
A condition is simply a callable that takes the context object as input and returns a boolean value indicating whether the condition is met or not.
This allows for dynamic control of the solver workflow based on the current state of the context object.


.. code-block:: python

    class Condition:

        def __init__(self, condition_func: Callable[..., bool], name: str = "Condition"):
            self._condition_func = condition_func
            self._name = name


The operators ``&``, ``|``, and ``~`` are overloaded to allow combining conditions using logical AND, OR, and NOT operations, respectively.
This enables the creation of complex conditions by combining simpler ones.

.. code-block:: python

    c1 = Condition( MaxIterations(5), "AlwaysTrue")
    c2 = Condition(lambda: check_something(), "MyCheck")
    combined = (c1 & c2) | ~c1


This also allows to modify the solver workflow and execution dynamically based on the current state of the context object.
Models can define their own conditions that can be used to modify the ``ConditionalOp`` and ``IterativeOp``s defined in the solver.
A common example for this would be a steady state solver that converged successfully and the residuals are below a certain threshold.


StepBuilder
^^^^^^^^^^^^^

It is possible to build required operations from just the operations defined in the solver and model classes with the available metadata in the Operations.

.. code-block:: python

    @Solver
    class IncompressibleFluidSolver:
        models: list[IncompressibleFluidModel]  # Additional physics models
        @Solver.operation(...)
        def momentum(self, ...): pass
        @Solver.operation(...)
        def continuity(self, ...): pass
        @Solver.operation(...)
        def update_turbulence(self, ...): pass


However, this approach is difficult to comprehend and a the solver workflow is not easily readable, especially for complex solvers and nested operations.
The classical approach in contrast is a lot easier to read and understand as the solver workflow is defined in a single method.

.. code-block:: python

    # pseudo code for a classical PIMPLE solver loop
    while runTime.loop():
        Info(f"Time = {runTime.timeName()}")

        # Compute Courant number
        cfl_number.setDelta(runTime, phi)

        while pimple.loop():
            UEqn = momentum_equation(...)

            while pimple.correct():
                pressure_correction(UEqn,...)

            if pimple.turbCorr():
                laminarTransport.correct()
                turbulence.correct()

        runTime.write(True)
        runTime.printExecutionTime()

    Info("End")


However, this classical approach lacks modularity and extensibility as the solver workflow is hardcoded in a single method.
To combine the advantages of both approaches, the ``StepBuilder`` class is introduced to build complex solver workflows in a more readable way.

The ``StepBuilder`` exposes two methods, ``step()`` and ``loop()``, plus context-manager support.
``step(op)`` appends a sequential operation; ``loop(op)`` appends an iterative operation **and returns a fresh ``StepBuilder``** scoped to the new loop's sub-operations.
Because ``__enter__`` returns ``self``, ``loop()`` can be used directly with ``with`` to nest scopes:

.. code-block:: python

    @incompressibleFluid.execution_graph_step
    def execution_graph(self, domain_name: str | None = None) -> tuple[StepBuilder, Operations]:
        ops = self.operations
        algorithm_model = self.state.core_models[0]
        algo_ops = Operations(algorithm_model._build_operations_for(algorithm_model))

        builder = StepBuilder()
        time_loop_op = Operation(
            func=IterativeOp(TimeLoop()),
            metadata=OperationMetadata(op_name="time_loop"),
        )
        with builder.loop(time_loop_op) as time_builder:
            time_builder.step(ops["set_time_step"])
            time_builder.step(ops["increment_time"])

            with time_builder.loop(algo_ops["inner_loop"]) as inner_builder:
                inner_builder.step(algo_ops["momentum"])
                inner_builder.step(algo_ops["continuity"])
                inner_builder.step(ops["turbulence_correction"])

            time_builder.step(ops["write_output"])

        # Collect operations contributed by optional models
        model_ops = Operations()
        for m in self.state.optional_models:
            model_ops.add(m.operations)

        return builder, model_ops

The solver returns the ``StepBuilder`` (which describes the *structure*) together with a flat ``Operations`` container of model contributions.
The two are merged by the ``DAGResolver``, which inspects each model operation's ``depends_on`` / ``before`` metadata, infers the correct loop scope, topologically sorts within each scope, and rebuilds the structure:

.. code-block:: python

    from neofoam.framework.graph import DAGResolver

    builder, model_ops = solver.execution_graph()
    resolver = DAGResolver()
    resolved = resolver.resolve(builder, model_ops)
    resolved.operations.run(ctx)

This split keeps the solver's loop topology declarative while letting models inject operations at runtime without the solver having to know about them.
