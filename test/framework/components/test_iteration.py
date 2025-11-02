import pytest
from typing import Literal
from pydantic import BaseModel
from foamadapter.framework.iteration import Iteration


@Iteration
class MyCustomIteration(BaseModel):
    name: Literal["MyCustomIteration"] = "MyCustomIteration"


    @Iteration.step(step_number=2, depends_on=["step_one"])
    def step_two(self):
        pass

    @Iteration.step(step_number=1) # check if the steps are sorted
    def step_one(self):
        pass

    @Iteration.step(step_number=3, depends_on=["step_two"])
    def step_three(self):
        pass

    @Iteration.step(step_number=4, depends_on=["step_three"])
    def step_four(self):
        pass


def test_iteration_steps_registration():

    assert MyCustomIteration.number_steps() == 4
    
    first_step = MyCustomIteration._steps[0]
    assert first_step.step_name == "step_one"
    assert first_step.step_number == 1
    assert first_step.depends_on == []
    assert first_step.func == MyCustomIteration.step_one

    second_step = MyCustomIteration._steps[1]
    assert second_step.step_name == "step_two"
    assert second_step.step_number == 2
    assert second_step.depends_on == ["step_one"]
    assert second_step.func == MyCustomIteration.step_two

    third_step = MyCustomIteration._steps[2]
    assert third_step.step_name == "step_three"
    assert third_step.step_number == 3
    assert third_step.depends_on == ["step_two"]
    assert third_step.func == MyCustomIteration.step_three

    fourth_step = MyCustomIteration._steps[3]
    assert fourth_step.step_name == "step_four"
    assert fourth_step.step_number == 4
    assert fourth_step.depends_on == ["step_three"]
    assert fourth_step.func == MyCustomIteration.step_four

    solver = MyCustomIteration()
    assert solver.name == "MyCustomIteration"

    schema = MyCustomIteration.model_json_schema()
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "_steps" not in schema["properties"]
    assert "_step_data" not in schema["properties"]
