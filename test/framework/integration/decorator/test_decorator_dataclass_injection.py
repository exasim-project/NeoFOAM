from dataclasses import dataclass

from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.decorator import operation
from foamadapter.framework.operations import (
    context_adapter,
)


@dataclass
class CombinedData:
    b: float
    c: float


class SomeClass:
    @operation
    def member_function(self, a: int, combined: CombinedData) -> FieldUpdates:
        a += 1
        combined.b += 2
        combined.c += 3.0
        return FieldUpdates({"a": a, "b": combined.b, "c": combined.c})


def test_context_adapter_step():
    """
    Tests the `context_adapter` decorator by verifying that it correctly
    injects the function parameters from a `Context` object and returns
    the a runnable function that updates the context fields.
    """
    sc = SomeClass()

    # create a new function that takes the context as input
    adapted_func = context_adapter(sc.member_function)

    ctx = Context(fields={"a": 0, "b": 0, "c": 0.0}, models={})
    # run the adapted function
    adapted_func(ctx)

    # verify that the context fields have been updated correctly
    assert ctx.fields["a"] == 1
    assert ctx.fields["b"] == 2
    assert ctx.fields["c"] == 3.0
