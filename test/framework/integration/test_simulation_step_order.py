from pathlib import Path

import pytest

from foamadapter.framework.dag import (
    NodeData,
    StepNumber,
    build_global_dag,
    compute_nodes_order,
)
from foamadapter.framework.operations import Operation, Operations, StepBuilder
from foamadapter.framework.pyvis_utils import digraph_to_pyvis_html
from foamadapter.framework.model import Model, ModelInterface
from foamadapter.framework.solver import Solver, Step
from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.dag import build_dag, compute_steps_order
from pydantic import BaseModel
from typing import Literal, Any
from foamadapter.framework.simulation import Simulation, Domain


@Solver
class MySolver(BaseModel):
    name: Literal["MySolver"] = "MySolver"
    param1: float
    models: list[Any]

    def create_context(self) -> Context:
        ctx = Context(fields={}, models={})
        ctx.fields["a"] = 0.0
        return ctx

    @Solver.step(step_number=1, depends_on=[])  # check if the steps are sorted
    def step_one(self, a: float):
        a += self.param1 + 1.0
        return FieldUpdates({"a": a})

    @Solver.step(step_number=2, depends_on=["step_one"])
    def step_two(self, a: float):
        a += self.param1 + 1.0
        return FieldUpdates({"a": a})

    @Solver.step(step_number=3, depends_on=["step_two"])
    def step_three(self, a: float):
        a += self.param1 + 1.0
        return FieldUpdates({"a": a})

    @Solver.step(step_number=4, depends_on=["step_three"])
    def step_four(self, a: float):
        a += self.param1 + 1.0
        return FieldUpdates({"a": a})

    def operations(self, domain_name: str | None = None) -> Operations:
        steps = [*self._steps]
        for step in steps:
            step.cls = self
            step.domain = domain_name
        for models in self.models:
            steps.extend(models.operations(domain_name=domain_name))
        
        ops = StepBuilder()

        # with ops as main_loop:
        #     ops.steps(Operation(,))

        
        return steps

    def dependencies(self, domain_name: str) -> list[NodeData]:
        nodedata = []
        for step in self.operations(domain_name=domain_name):
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
    name: str

    @Model.step(step_number=1, depends_on=["step_one"])
    def initialize(self, a: float):
        a += self.param + 1.0
        return FieldUpdates({"a": a})

    @Model.step(step_number=2, depends_on=["step_three"])
    def process(self, a: float):
        a += self.param + 1.0
        return FieldUpdates({"a": a})

    def operations(self, domain_name: str | None = None) -> list[Step]:
        steps = [*self._steps]
        for step in steps:
            step.cls = self
            step.domain = domain_name
        return steps

    def dependencies(self, domain_name: str | None = None) -> list[NodeData]:

        nodedata = []
        for step in self.operations(domain_name=domain_name):
            nodedata.append(
                NodeData(
                    name=step.name,
                    depends_on=step.dependency_names,
                    shape="box",
                    step_number=StepNumber(f"{step.step_number}.0.0"),
                )
            )
        return nodedata

@pytest.mark.skip(reason="Failing test, needs investigation")
def test_simulation_step_order():
    model1 = MyModel(param=2.0, name="my_model1")
    sim = Simulation(
        domains=[
            Domain(name="region1", solver=MySolver(param1=1.0, models=[model1])),
        ],
        coupling_interface=[],
    )

    #
    steps = sim.domains[0].steps()
    nodes = sim.domains[0].dependencies()
    nodes_ordered = compute_nodes_order(nodes)
    steps_ordered = compute_steps_order(steps, nodes)

    assert len(steps_ordered) == 6
    assert nodes_ordered == [
        "region1.step_one",
        "region1.initialize",
        "region1.step_two",
        "region1.step_three",
        "region1.process",
        "region1.step_four",
    ]

    if False:
        dag = sim.dependency_graph()
        parent_dir = Path(__file__).parent
        digraph_to_pyvis_html(
            dag, html_path=str(parent_dir / "dag_sim_step_order.html")
        )

@pytest.mark.skip(reason="Failing test, needs investigation")
def test_simulation_run():
    model1 = MyModel(param=2.0, name="my_model1")
    sim = Simulation(
        domains=[
            Domain(name="region1", solver=MySolver(param1=1.0, models=[model1])),
        ],
        coupling_interface=[],
    )

    ctx = sim.init_simulation_context()
    sim.main_loop(ctx)
    fields = ctx.domain_context["region1"].fields
    assert fields["a"] == 8.0 + 6.0
