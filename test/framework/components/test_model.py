from typing import Literal

from pydantic import BaseModel

from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.model import Model, ModelInterface
from foamadapter.framework.operations import Operation, OperationCollection


@Model
class MyCustomModel(BaseModel):
    name: Literal["MyCustomModel"] = "MyCustomModel"

    @Model.step(step_number=1)  # check if the steps are sorted
    def step_one(self):
        pass

    @Model.step(step_number=2, depends_on=["step_one"])
    def step_two(self):
        pass

    @Model.step(step_number=3, depends_on=["step_two"])
    def step_three(self):
        pass

    @Model.step(step_number=4, depends_on=["step_three"])
    def step_four(self):
        pass

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops

    def run(self, ctx):
        pass


def test_model_steps_registration():
    assert issubclass(MyCustomModel, ModelInterface)
    # assert MyCustomModel.number_steps() == 4

    my_model = MyCustomModel()
    ops = my_model.operations()
    assert len(ops) == 4

    first_step = ops[0]
    assert first_step.step_name == "step_one"
    assert first_step.step_number == 1
    assert first_step.depends_on == []

    second_step = ops[1]
    assert second_step.step_name == "step_two"
    assert second_step.step_number == 2
    assert second_step.depends_on == ["step_one"]

    third_step = ops[2]
    assert third_step.step_name == "step_three"
    assert third_step.step_number == 3
    assert third_step.depends_on == ["step_two"]

    fourth_step = ops[3]
    assert fourth_step.step_name == "step_four"
    assert fourth_step.step_number == 4
    assert fourth_step.depends_on == ["step_three"]

    solver = MyCustomModel()
    assert solver.name == "MyCustomModel"

    schema = MyCustomModel.model_json_schema()
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "_steps" not in schema["properties"]
    assert "_step_data" not in schema["properties"]
