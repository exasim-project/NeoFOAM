from foamadapter.framework.solver import Solver
from foamadapter.framework.context import Context, Model, Field, FieldUpdates
from typing import Annotated
from dataclasses import dataclass

@dataclass
class Data:
    a: str
    b: float
    m_a: Model[str]


def test_injection_dataclass():
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = "a"
    ctx.fields["b"] = 1.0
    ctx.models["m_a"] = "m_a"

    @Solver.step(step_number=1)
    def function(data: Data) -> str:
        """A test function that asserts and returns."""
        assert data.a == "a"
        assert data.b == 1.0
        assert data.m_a == "m_a"
        return FieldUpdates({"a": data.a})
    
    function(data=Data(a="a",b=1.0,m_a="m_a"))
    function.run(ctx)


def test_injection_dataclass_with_other_params():
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = "a"
    ctx.fields["b"] = 1.0
    ctx.models["m_a"] = "m_a"
    ctx.fields["c"] = 1

    @Solver.step(step_number=1)
    def function(data: Data, c: int) -> str:
        """A test function that asserts and returns."""
        assert data.a == "a"
        assert data.b == 1.0
        assert c == 1
        assert data.m_a == "m_a"
        return FieldUpdates({"a": data.a})
    
    function(data=Data(a="a",b=1.0,m_a="m_a"), c= 1)
    function.run(ctx)