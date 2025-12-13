from typing import Literal

import pytest
from pydantic import BaseModel

from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.dag import compute_nodes_order, compute_steps_order
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.model import Model
from foamadapter.framework.operations import Operation, OperationCollection
from foamadapter.framework.solver import Solver


@Solver
class MySolver(BaseModel):
    name: Literal["MySolver"] = "MySolver"
    param1: float

    @Solver.operation(operation_number=1, depends_on=[])  # check if the ops are sorted
    def op_one(self, a: float) -> FieldUpdates:
        return FieldUpdates(a=self.param1 + 1.0)

    @Solver.operation(operation_number=2, depends_on=["op_one"])
    def op_two(self, a: float) -> FieldUpdates:
        return FieldUpdates(a=a + self.param1 + 1.0)

    @Solver.operation(operation_number=3, depends_on=["op_two"])
    def op_three(self, a: float) -> FieldUpdates:
        return FieldUpdates(a=a + self.param1 + 1.0)

    @Solver.operation(operation_number=4, depends_on=["op_three"])
    def op_four(self, a: float) -> FieldUpdates:
        return FieldUpdates(a=a + self.param1 + 1.0)

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops

    def main_loop(self, ctx: Context): ...


@Model
class MyModel(BaseModel):
    param: float

    @Model.operation(operation_number=1, depends_on=["op_one"])
    def initialize(self, a: float) -> FieldUpdates:
        return FieldUpdates(a=a + self.param + 1.0)

    @Model.operation(operation_number=2, depends_on=["op_three"])
    def process(self, a: float) -> FieldUpdates:
        return FieldUpdates(a=a + self.param + 1.0)

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops


def test_run_steps():
    solver1 = MySolver(param1=1.0)
    model1 = MyModel(param=1.0)

    steps_solver1 = solver1.operations()
    steps_model1 = model1.operations()
    assert len(steps_solver1) == 4
    assert len(steps_model1) == 2

    steps_solver1.add(steps_model1)
    # steps = steps_model1 + steps_solver1
    model1.param = 2.0
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = 0.0
    for op in steps_solver1:
        op.run(ctx)
    assert ctx.fields["a"] == 8.0 + 6.0


@pytest.mark.skip(reason="Not implemented/updated yet")
def test_step_order():
    solver1 = MySolver(param1=1.0)
    model1 = MyModel(param=1.0)

    steps_solver1 = solver1.operations("domain1")
    steps_model1 = model1.operations("domain1")
    assert len(steps_solver1) == 4
    assert len(steps_model1) == 2

    steps = steps_model1 + steps_solver1
    nodes = model1.dependencies("domain1") + solver1.dependencies("domain1")

    nodes_ordered = compute_nodes_order(nodes)
    steps_ordered = compute_steps_order(steps, nodes)

    assert nodes_ordered == [step.name for step in steps_ordered]
    a = 0
    for step in steps_ordered:
        a += step()
    assert a == 8.0 + 4.0
