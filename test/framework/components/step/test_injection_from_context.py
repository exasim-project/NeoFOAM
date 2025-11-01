from foamadapter.framework.step import step
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

    class SomeClass:

        def __init__(self):
            self.param = 10

        @step(step_number=1)
        def function(self, data: Data) -> str:
            """A test function that asserts and returns."""
            data.a = "a"
            data.b = 1.0 + self.param
            data.m_a = "m_a"
            return FieldUpdates({"a": data.a, "b": data.b, "m_a": data.m_a})

    sc = SomeClass()
    res_func = sc.function(data=Data(a="a", b=1.0, m_a="m_a"))
    res_ctx = sc.function.run(sc, ctx)

    assert res_func == res_ctx


def test_injection_dataclass_with_other_params():
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = "a"
    ctx.fields["b"] = 1.0
    ctx.models["m_a"] = "m_a"
    ctx.fields["c"] = 1

    class SomeClass:

        def __init__(self):
            self.param = 10

        @step(step_number=1)
        def function(self, data: Data, c: int) -> str:
            """A test function that asserts and returns."""
            data.a = "a"
            data.b = 1.0 + self.param
            c = 1
            data.m_a = "m_a"
            return FieldUpdates({"a": data.a, "b": data.b, "m_a": data.m_a, "c": c})

    sc = SomeClass()
    res_func = sc.function(data=Data(a="a", b=11.0, m_a="m_a"), c=1)
    res_ctx = sc.function.run(sc, ctx)
    assert res_func == res_ctx
