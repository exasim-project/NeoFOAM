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

from foamadapter.framework.step import step
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
        >>> @step(step_number=1)
        ... def my_step(value: int) -> FieldUpdates:
        ...     return FieldUpdates({"result": value * 2})
        >>> 
        >>> my_step.run(ctx)  # Automatically injects value=42
    """
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = "a"
    ctx.fields["b"] = 1.0

    class SomeClass:

        def __init__(self):
            self.param = 10

        @step(step_number=1)
        def function(self, a: str, b: float) -> str:
            """A test function that demonstrates basic injection."""
            a = "a"
            b = 1.0 + self.param
            return FieldUpdates({"a": a, "b": b})
    
    # Test both direct call and context injection

    sc = SomeClass()

    res_func = sc.function(a="a", b = 1.0)
    res_ctx = sc.function.run(sc,ctx)

    assert res_func == res_ctx

def test_injection_model():
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = "a"
    ctx.fields["b"] = 1.0

    ctx.models["c"] = 1.0

    class SomeClass:
        

        @step(step_number=1)
        def function1(self,a: str, c: Model[float]) -> str:
            """A test function that asserts and returns."""
            assert a == "a"
            assert c == 1.0
            return FieldUpdates({"a": a})
        

        @step(step_number=2)
        def function2(self, a: Field[str], c: Model[float]) -> str:
            """A test function that asserts and returns."""
            assert a == "a"
            assert c == 1.0
            return FieldUpdates({"a": a})
    
    sc = SomeClass()

    
    res1 = sc.function1(a="a", c = 1.0)
    res1_ctx = sc.function1.run(sc, ctx)
    assert res1 == res1_ctx

    res2 = sc.function2(a="a", c = 1.0)
    assert res2 == sc.function2.run(sc, ctx)

    res3 = SomeClass.function2(sc, a="a", c = 1.0)
    assert res2 == res3
    assert res2 == SomeClass.function2.run(sc, ctx)

@Solver
class MySolver():

    def __init__(self, init_value=1):
        self.init_value = init_value

    @Solver.step(step_number=1)
    def init(self,a: float):
        a = self.init_value
        return FieldUpdates({"a": a})

    @Solver.step(step_number=2)
    def factor2(self,a: float):
        a = a*2
        return FieldUpdates({"a": a})

    @Solver.step(step_number=3)
    def add5(self,a: float):
        a = a + 5
        return FieldUpdates({"a": a})


def test_solver():
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = 0

    solver = MySolver()

    for step in solver._steps:
        step.cls = solver  # Bind instance to step
        step.run(ctx)

    assert ctx.fields["a"] == 7

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
                step.cls = self.solver  # Bind instance to step
                step.run(ctx)




def test_runner():
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = 0

    runner = Runner(solver=MySolver())

    runner.run(ctx=ctx)

    assert ctx.fields["a"] == 7