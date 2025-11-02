from pathlib import Path

from foamadapter.framework.dag import NodeData, StepNumber, build_global_dag, compute_nodes_order
from foamadapter.framework.pyvis_utils import digraph_to_pyvis_html
from foamadapter.framework.model import Model
from foamadapter.framework.solver import Solver, Step
from foamadapter.framework.context import Context
from foamadapter.framework.dag import build_dag, compute_steps_order
from pydantic import BaseModel
from typing import Literal
import functools


@Solver
class MySolver(BaseModel):
    name: Literal["MySolver"] = "MySolver"
    param1: float
    

    @Solver.step(step_number=1, depends_on=[])  # check if the steps are sorted
    def step_one(self):
        return self.param1 + 1.0

    @Solver.step(step_number=2, depends_on=["step_one"])
    def step_two(self):
        return self.param1 + 1.0

    @Solver.step(step_number=3, depends_on=["step_two"])
    def step_three(self):
        return self.param1 + 1.0

    @Solver.step(step_number=4, depends_on=["step_three"])
    def step_four(self):
        return self.param1 + 1.0

    def steps(self, domain_name: str | None = None) -> list[Step]:
        steps = [*self._steps]
        for step in steps:
            step.cls = self
            step.domain = domain_name
        return steps

    def dependencies(self, domain_name: str) -> list[NodeData]:
        nodedata = []
        for step in self.steps(domain_name=domain_name):
            nodedata.append(
                NodeData(
                    name=step.name,
                    depends_on=step.dependency_names,
                    shape="box",
                    step_number=StepNumber(f"{step.step_number}.0.0"),
                )
            )
        return nodedata

    def main_loop(self, ctx: Context): ...

@Model
class MyModel(BaseModel):
    param: float

    @Model.step(step_number=1, depends_on=["step_one"])
    def initialize(self):
        return self.param + 1.0

    @Model.step(step_number=2, depends_on=["step_three"])
    def process(self):
        return self.param + 1.0

    def steps(self, domain_name: str | None = None) -> list[Step]:
        steps = [*self._steps]
        for step in steps:
            step.cls = self
            step.domain = domain_name
        return steps
    
    def dependencies(self, domain_name: str | None = None) -> list[NodeData]:
        nodedata = []
        for step in self.steps(domain_name=domain_name):
            nodedata.append(
                NodeData(
                    name=step.name,
                    depends_on=step.dependency_names,
                    shape="box",
                    step_number=StepNumber(f"{step.step_number}.0.0"),
                )
            )
        return nodedata



def test_run_steps():
    solver1 = MySolver(param1=1.0)
    model1 = MyModel(param=1.0)

    steps_solver1 = solver1.steps()
    steps_model1 = model1.steps()
    assert len(steps_solver1) == 4
    assert len(steps_model1) == 2

    steps = steps_model1 + steps_solver1
    model1.param = 2.0
    a = 0
    for step in steps:
        a += step()
    assert a == 8.0 + 6.0

def test_step_order():
    solver1 = MySolver(param1=1.0)
    model1 = MyModel(param=1.0)

    steps_solver1 = solver1.steps("domain1")
    steps_model1 = model1.steps("domain1")
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

