Main Components
===============


As introduced in the architecture overview, the FoamAdapter solver framework tries to make the solver more modular and easier to extend by breaking down the solver into smaller components.



Main Concepts
^^^^^^^^^^^^^

The main concept of the FoamAdapter solver framework is to represent a CFD solver a series of operations that update fields based on governing equations.

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
        runTime: Any = None


Each operation is a class that implements a specific functionality, such as updating a field based on a governing equation or applying boundary conditions.
It can be created by using the ``@Solver.step`` or ``@Model.step`` decorator.
The decorator method flags the bounded method as a step and collects metadata such as step name, step number, and dependencies.


.. code-block:: python

    @Solver.step
    def solve_momentum(self, U, p) -> FieldUpdates:
        # ... computation
        return FieldUpdates({"U": U_new})

Additionally, decorated methods are self-contained and the dependencies are injected automatically by the framework by inspecting the method signature.
The framework uses the method signature to determine which fields or models are required for the operation and retrieves them from the context object.
Consequently, the user only needs to specify the required inputs as method arguments, and the framework takes care of providing the correct data when the operation is executed.
Another advantage of this approach is that it makes the operations more modular and easier to test, as they can be executed independently with different inputs.
It improves code readability and maintainability by clearly defining the dependencies of each operation.
The dependencies between the operations can also be visualized which helps to understand the overall structure of the solver.


Operations
^^^^^^^^^^

There are three main types of operations in the framework:

* ``SequentialOp`` represents a regular step in the solver sequence and contains a single function to execute.
* ``ConditionalOp`` represents a step that is executed only if a certain condition is met, and has multiple sub-operations.
* ``IterativeOp`` represents a step that is executed repeatedly while a certain condition is met, and has multiple sub-operations.

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


All operations stored as instances of the ``Operation`` class, which contains the following attributes:

.. code-block:: python

    @dataclass
    class Operation:
        """A concrete step class that wraps a function with metadata."""

        func: Union[ConditionalOp, IterativeOp, SequentialOp]
        step_number: StepNumber = None
        step_name: str = None
        domain_name: str | None = None
        depends_on: list[str] | None = None
        shape: str = "box"
        color: str = "lightblue"
        level: int = 0
        sub_steps: list["Operation"] = field(default_factory=list)

The solver framework gathers all ``Operation`` instances defined in the solver and model classes and constructs a workflow that can be executed in sequence.
The resulting workflow is represented by the ``Operations`` class that is a container for all operations in the solver:

.. code-block:: python

    class Operations:
        def __init__(self, operations: list[Operation] = None):
            if operations is not None:
                self.ops = operations
            else:
                self.ops: list[Operation] = []

        def run(self, ctx):
        for operation in self.ops:
            operation.run(ctx)

It provides a ``run`` method that executes all operations in sequence, passing the context object to each operation.


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
A common example for this would be a steady state solver that converged sucessfully and the residuals are below a certain threshold.


StepBuilder
^^^^^^^^^^^^^

It is possible to build required operations from just the operations defined in the solver and model classes with the availble metadata in the Operations.

.. code-block:: python

    @Solver
    class IncompressibleFluidSolver:
        models: list[IncompressibleFluidModel]  # Additional physics models
        @Solver.step(...)
        def momentum(self, ...): pass
        @Solver.step(...)
        def continuity(self, ...): pass
        @Solver.step(...)
        def update_turbulence(self, ...): pass


However, this approach is difficult to comprehend and a the solver workflow is not easily readable, exspecially for complex solvers and nested operations.
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
To combine the advantages of both approaches, the ``StepBuilder`` class is introduced to build complex solver workflows in a more readable and understandable way.
The ``StepBuilder`` class provides a fluent interface to define the solver workflow by chaining method calls.
This makes it easier to read and understand the solver

It is used in the define_operations method of the solver class to build the solver workflow.

.. code-block:: python

    def define_operations(self, domain_name: str | None = None) -> Operations:
        ops_col = self.operations(domain_name) # list of all available operations including the models
        op_build = StepBuilder()

        with op_build as main_loop:
            main_loop.step(ops_col["cfl_number"])

            with main_loop.loop(ops_col["loop_condition"]) as pimple:
                pimple.step(ops_col["momentum_equation"])

                pimple.step(ops_col["pressure_correction"])

                pimple.step(ops_col["turbulence_model"])

        op_build.update_operations(ops_col) # add or modify steps defined by the models of a solver

        return op_build.operations

