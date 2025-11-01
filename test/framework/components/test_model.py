import pytest
from typing import Literal
from pydantic import BaseModel
from foamadapter.framework.model import Model


@Model
class MyCustomModel(BaseModel):
    name: Literal["MyCustomModel"] = "MyCustomModel"


    @Model.step(step_number=2)
    def step_two(self):
        pass

    @Model.step(step_number=1) # check if the steps are sorted
    def step_one(self):
        pass

    @Model.step(step_number=3)
    def step_three(self):
        pass

    @Model.step(step_number=4)
    def step_four(self):
        pass


def test_model_steps_registration():

    assert MyCustomModel.number_steps() == 4
    
    first_step = MyCustomModel._steps[0]
    assert first_step.step_name == "step_one"
    assert first_step.step_number == 1
    assert first_step.depends_on == []
    assert first_step.func == MyCustomModel.step_one

    second_step = MyCustomModel._steps[1]
    assert second_step.step_name == "step_two"
    assert second_step.step_number == 2
    assert second_step.depends_on == ["step_one"]
    assert second_step.func == MyCustomModel.step_two

    third_step = MyCustomModel._steps[2]
    assert third_step.step_name == "step_three"
    assert third_step.step_number == 3
    assert third_step.depends_on == ["step_two"]
    assert third_step.func == MyCustomModel.step_three

    fourth_step = MyCustomModel._steps[3]
    assert fourth_step.step_name == "step_four"
    assert fourth_step.step_number == 4
    assert fourth_step.depends_on == ["step_three"]
    assert fourth_step.func == MyCustomModel.step_four

    solver = MyCustomModel()
    assert solver.name == "MyCustomModel"

    schema = MyCustomModel.model_json_schema()
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "_steps" not in schema["properties"]
    assert "_step_data" not in schema["properties"]
