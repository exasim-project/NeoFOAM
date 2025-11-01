Testing Framework
=================

This section demonstrates the FoamAdapter framework through executable tests that serve as both documentation and validation.

Framework Tests
---------------

Solver Injection Tests
~~~~~~~~~~~~~~~~~~~~~~

The framework supports dependency injection for solver steps. Here's how it works:

.. testcode::

    from foamadapter.framework.solver import Solver
    from foamadapter.framework.context import Context, Model, Field, FieldUpdates

    # Create a context with fields and models
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = "a"
    ctx.fields["b"] = 1.0

    @Solver.step(step_number=1)
    def function(a: str, b: float) -> str:
        """A test function that demonstrates injection."""
        print(f"Inside function: a = {a!r}, b = {b!r}")
        assert a == "a"
        assert b == 1.0
        return FieldUpdates({"a": a})
    
    # Test direct call
    result = function(a="a", b=1.0)
    
    # Test context injection
    function.run(ctx)

.. testoutput::

    Inside function: a = 'a', b = 1.0

Model and Field Injection
~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework also supports Model and Field type annotations:

.. testcode::

    ctx = Context(fields={}, models={})
    ctx.fields["a"] = "a"
    ctx.fields["b"] = 1.0
    ctx.models["c"] = 1.0

    @Solver.step(step_number=1)
    def function_with_model(a: str, c: Model[float]) -> str:
        """Function using Model injection."""
        assert a == "a"
        assert c == 1.0
        return FieldUpdates({"a": a})
    
    @Solver.step(step_number=2)
    def function_with_field(a: Field[str], c: Model[float]) -> str:
        """Function using Field injection."""
        assert a == "a"
        assert c == 1.0
        return FieldUpdates({"a": a})
    
    # Test both functions
    function_with_model(a="a", c=1.0)
    function_with_model.run(ctx)
    
    function_with_field(a="a", c=1.0)
    function_with_field.run(ctx)
    
    print("Model and Field injection working correctly")

.. testoutput::

    Model and Field injection working correctly

Solver Class Example
~~~~~~~~~~~~~~~~~~~~

You can also define solvers as classes with multiple steps:

.. testcode::

    from dataclasses import dataclass

    @Solver
    class MySolver():

        @Solver.step(step_number=1)
        def init(a: float):
            a = 1
            return FieldUpdates({"a": a})

        @Solver.step(step_number=2)
        def factor2(a: float):
            a = a * 2
            return FieldUpdates({"a": a})

        @Solver.step(step_number=3)
        def add5(a: float):
            a = a + 5
            return FieldUpdates({"a": a})

    class Condition:
        def __init__(self, max_iteration: float):
            self.max_iteration = max_iteration
            self.iteration = 0

        def __call__(self, *args, **kwds) -> bool:
            self.iteration += 1
            return self.iteration < self.max_iteration

    @dataclass
    class Runner():
        solver: MySolver

        def run(self, ctx: Context):
            condition = Condition(max_iteration=5)
            while condition():
                for step in self.solver._steps:
                    step.run(ctx)

Now let's test the complete solver pipeline:

.. testcode::

    # Initialize context
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = 0

    # Create and run solver
    runner = Runner(solver=MySolver())
    runner.run(ctx=ctx)

    print(f"Final result: {ctx.fields['a']}")
    assert ctx.fields["a"] == 7

.. testoutput::

    Final result: 7

This demonstrates how the solver processes through multiple steps:

1. **init**: Sets a = 1
2. **factor2**: Multiplies by 2 (a = 2) 
3. **add5**: Adds 5 (a = 7)
4. Repeats 5 times, but since step 1 resets to 1 each time, final result is 7

Running Tests
~~~~~~~~~~~~~

To run these documentation tests, use:

.. code-block:: bash

    # Run doctests only
    make doctest
    
    # Or with sphinx-build directly
    sphinx-build -b doctest . _build/doctest