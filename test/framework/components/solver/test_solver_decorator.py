import pytest
from typing import Literal
from pydantic import BaseModel
from foamadapter.framework.solver import Solver


@Solver
class MyCustomSolver(BaseModel):
    name: Literal["MyCustomSolver"] = "MyCustomSolver"


    @Solver.step(step_number=2)
    def step_two(self):
        pass

    @Solver.step(step_number=1) # check if the steps are sorted
    def step_one(self):
        pass

    @Solver.step(step_number=3)
    def step_three(self):
        pass

    @Solver.step(step_number=4)
    def step_four(self):
        pass


def test_solver_steps_registration():

    assert MyCustomSolver.number_steps() == 4
    
    first_step = MyCustomSolver._steps[0]
    assert first_step.step_name == "step_one"
    assert first_step.step_number == 1
    assert first_step.depends_on == []
    assert first_step.func == MyCustomSolver.step_one

    second_step = MyCustomSolver._steps[1]
    assert second_step.step_name == "step_two"
    assert second_step.step_number == 2
    assert second_step.depends_on == ["step_one"]
    assert second_step.func == MyCustomSolver.step_two

    third_step = MyCustomSolver._steps[2]
    assert third_step.step_name == "step_three"
    assert third_step.step_number == 3
    assert third_step.depends_on == ["step_two"]
    assert third_step.func == MyCustomSolver.step_three

    fourth_step = MyCustomSolver._steps[3]
    assert fourth_step.step_name == "step_four"
    assert fourth_step.step_number == 4
    assert fourth_step.depends_on == ["step_three"]
    assert fourth_step.func == MyCustomSolver.step_four

    solver = MyCustomSolver()
    assert solver.name == "MyCustomSolver"

    schema = MyCustomSolver.model_json_schema()
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "_steps" not in schema["properties"]
    assert "_step_data" not in schema["properties"]



def test_solver_step_requires_step_number():
    # Define a class with a step missing step_number
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'step_number'"):
        @Solver
        class BadSolver(BaseModel):
            @Solver.step()
            def step_without_number(self):
                pass