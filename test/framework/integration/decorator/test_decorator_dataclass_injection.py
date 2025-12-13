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
        combined.b += 2
        combined.c += 3.0
        return FieldUpdates({"b": combined.b, "c": combined.c})


def test_context_adapter_step():
    sc = SomeClass()

    adapted_func = context_adapter(sc.member_function)

    ctx = Context(fields={"a": 0, "b": 0, "c": 0.0}, models={})
    adapted_func(ctx)

    assert ctx.fields["b"] == 2
    assert ctx.fields["c"] == 3.0
