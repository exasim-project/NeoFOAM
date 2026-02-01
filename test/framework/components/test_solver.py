import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - tests incomplete Protocol implementations"
)

from typing import Literal

from pydantic import BaseModel

from foamadapter.framework.context import Context
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.operations import Operation, OperationCollection
from foamadapter.framework.bkp_solver import Solver, SolverInterface


@Solver
class MyCustomSolver(BaseModel):
    name: Literal["MyCustomSolver"] = "MyCustomSolver"

    @Solver.operation(operation_number=1)
    def step_one(self):
        pass

    @Solver.operation(operation_number=2, depends_on=["step_one"])
    def step_two(self):
        pass

    @Solver.operation(operation_number=3, depends_on=["step_two"])
    def step_three(self):
        pass

    @Solver.operation(operation_number=4, depends_on=["step_three"])
    def step_four(self):
        pass

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops

    def main_loop(self, ctx: Context): ...


def test_solver_steps_registration():
    si: SolverInterface = MyCustomSolver()
    assert issubclass(MyCustomSolver, SolverInterface)
    # assert MyCustomSolver.number_steps() == 4

    ops = si.operations()
    assert len(ops) == 4

    first_step = ops[0]
    assert first_step.operation_name == "step_one"
    assert first_step.operation_number == 1
    assert first_step.depends_on == []

    second_step = ops[1]
    assert second_step.operation_name == "step_two"
    assert second_step.operation_number == 2
    assert second_step.depends_on == ["step_one"]

    third_step = ops[2]
    assert third_step.operation_name == "step_three"
    assert third_step.operation_number == 3
    assert third_step.depends_on == ["step_two"]

    fourth_step = ops[3]
    assert fourth_step.operation_name == "step_four"
    assert fourth_step.operation_number == 4
    assert fourth_step.depends_on == ["step_three"]

    solver = MyCustomSolver()
    assert solver.name == "MyCustomSolver"

    schema = MyCustomSolver.model_json_schema()
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "_steps" not in schema["properties"]
    assert "_step_data" not in schema["properties"]
