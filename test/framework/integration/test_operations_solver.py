import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - uses old @Solver/@Model decorators from bkp_solver.py. New FastAPI-like syntax tested in dummy_solver tests."
)

from typing import Literal

from pydantic import BaseModel

from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.model import Model
from foamadapter.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    Operations,
    StepBuilder,
)
from foamadapter.framework.bkp_solver import Solver


class MaxIterations:
    def __init__(self, max_iters=5):
        self.max_iters = max_iters
        self.current_iter = 0

    def __call__(self, ctx):
        self.current_iter += 1
        if self.current_iter <= self.max_iters:
            return True
        return False


@Model
class MyModel1(BaseModel):
    param1: float

    @Model.operation(operation_number=1, depends_on=["step1"])
    def model1_step_one(self, a: float):
        a += self.param1 + 1.0
        return FieldUpdates({"a": a})

    @Model.operation(operation_number=2, depends_on=["step2"])
    def model1_step_two(self, a: float):
        a += self.param1 + 1.0
        return FieldUpdates({"a": a})

    def operations(self) -> OperationCollection:
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops


@Solver
class FirstSolver(BaseModel):
    name: Literal["FirstSolver"] = "FirstSolver"
    model_1: MyModel1 = None

    def create_context(self) -> Context:
        ctx = Context(fields={}, models={})
        ctx.fields["a"] = 0.0
        return ctx

    @Solver.operation(operation_number=1)
    def init(self, a: float):
        a = 1
        return FieldUpdates({"a": a})

    @Solver.operation(operation_number=2, depends_on=["init"])
    def factor2(self, a: float):
        a = a * 2
        return FieldUpdates({"a": a})

    @Solver.operation(operation_number=3, depends_on=["factor2"])
    def add2(self, a: float):
        a = a + 2
        return FieldUpdates({"a": a})

    @Solver.operation(operation_number=4, depends_on=["add2"])
    def add5(self, a: float):
        a = a + 5
        return FieldUpdates({"a": a})

    def operations(self) -> OperationCollection:
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        if self.model_1 is not None:
            ops.add(self.model_1.operations())
        return ops

    def main_loop(self):
        ops = self.operations()

        main_loop = StepBuilder()

        with main_loop.loop(
            Operation(
                func=IterativeOp(MaxIterations(max_iters=4)),
                operation_name="loop_increment",
                operation_number=1,
            )
        ) as loop:
            loop.step(ops["init"])
            loop.step(ops["factor2"])
            loop.step(ops["add2"])
            loop.step(ops["add5"])

        if self.model_1 is not None:
            main_loop.step(ops["model1_step_one"])
            main_loop.step(ops["model1_step_two"])

        return main_loop.operations


def test_operations_solver():
    solver = FirstSolver()
    ctx = solver.create_context()
    ops_col = solver.operations()
    for op in ops_col:
        op.run(ctx)

    assert ops_col[0].operation_name == "init"
    assert ops_col[1].operation_name == "factor2"
    assert ops_col[2].operation_name == "add2"
    assert ops_col[3].operation_name == "add5"

    assert ctx.fields["a"] == 1 * 2 + 2 + 5

    ctx.fields["a"] = 0.0
    ops = Operations(ops_col)
    ops.run(ctx)
    assert ctx.fields["a"] == 1 * 2 + 2 + 5

    ctx.fields["a"] = 0.0
    main_loop_ops = solver.main_loop()
    main_loop_ops.run(ctx)
    assert ctx.fields["a"] == (1 * 2 + 2 + 5) * 1  # init resets each loop


def test_solver_with_model_operations():
    model = MyModel1(param1=3.0)
    solver = FirstSolver(model_1=model)
    ctx = solver.create_context()
    ops_col = solver.operations()
    for op in ops_col:
        op.run(ctx)
    # init: a=1, factor2: a=2, add2: a=4, add5: a=9, model1_step_one: a=13, model1_step_two: a=17
    assert ctx.fields["a"] == 1 * 2 + 2 + 5 + (3.0 + 1.0) + (3.0 + 1.0)

    ctx.fields["a"] = 0.0
    ops = Operations(ops_col)
    ops.run(ctx)
    assert ctx.fields["a"] == 1 * 2 + 2 + 5 + (3.0 + 1.0) + (3.0 + 1.0)

    ctx.fields["a"] = 0.0
    main_loop_ops = solver.main_loop()
    main_loop_ops.run(ctx)
    assert (
        ctx.fields["a"] == (1 * 2 + 2 + 5 + (3.0 + 1.0) + (3.0 + 1.0)) * 1
    )  # init resets each loop
