"""
Framework Tests: Dependency Injection and Solver Patterns
==========================================================

This module demonstrates and tests the FoamAdapter framework's core features:

- Dependency injection for solver steps
- Context-based field and model management  
- Solver class definitions with multiple steps
- Complete solver pipeline execution

These tests serve as both validation and documentation examples.
"""

from foamadapter.framework.solver import Solver
from foamadapter.framework.context import Context, Model, Field, FieldUpdates
from typing import Annotated
from dataclasses import dataclass


def test_injection():
    """
    Test basic dependency injection for solver steps.
    
    This example shows how the framework can automatically resolve
    dependencies from a Context object and inject them into solver
    step functions based on their type annotations.
    
    Example:
        >>> from foamadapter.framework.solver import Solver
        >>> from foamadapter.framework.context import Context, FieldUpdates
        >>> 
        >>> ctx = Context(fields={}, models={})
        >>> ctx.fields["value"] = 42
        >>> 
        >>> @Solver.step(step_number=1)
        ... def my_step(value: int) -> FieldUpdates:
        ...     return FieldUpdates({"result": value * 2})
        >>> 
        >>> my_step.run(ctx)  # Automatically injects value=42
    """
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = "a"
    ctx.fields["b"] = 1.0

    @Solver.step(step_number=1)
    def function(a: str, b: float) -> str:
        """A test function that demonstrates basic injection."""
        print(f"  Inside function: a = {a!r}, b = {b!r}")
        assert a == "a"
        assert b == 1.0
        return FieldUpdates({"a": a})
    
    # Test both direct call and context injection
    function(a="a", b = 1.0)
    function.run(ctx)

def test_injection_model():
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = "a"
    ctx.fields["b"] = 1.0

    ctx.models["c"] = 1.0

    @Solver.step(step_number=1)
    def function(a: str, c: Model[float]) -> str:
        """A test function that asserts and returns."""
        assert a == "a"
        assert c == 1.0
        return FieldUpdates({"a": a})
    
    function(a="a", c = 1.0)
    function.run(ctx)

    @Solver.step(step_number=1)
    def function2(a: Field[str], c: Model[float]) -> str:
        """A test function that asserts and returns."""
        assert a == "a"
        assert c == 1.0
        return FieldUpdates({"a": a})
    
    function2(a="a", c = 1.0)
    function2.run(ctx)

@Solver
class MySolver():

    @Solver.step(step_number=1)
    def init(a: float):
        a = 1
        return FieldUpdates({"a": a})

    @Solver.step(step_number=2)
    def factor2(a: float):
        a = a*2
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




def test_runner():
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = 0

    runner = Runner(solver=MySolver())

    runner.run(ctx=ctx)

    assert ctx.fields["a"] == 7