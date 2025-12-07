# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.dag import compute_nodes_order, compute_steps_order
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.model import Model, ModelInterface
from foamadapter.framework.operations import (
    Operation,
    OperationCollection,
    Operations,
    StepBuilder,
)
from foamadapter.framework.pyvis_utils import digraph_to_pyvis_html
from foamadapter.framework.simulation import Domain, Simulation
from foamadapter.framework.solver import Solver


@Solver
class MySolver(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: Literal["MySolver"] = "MySolver"
    param1: float
    models: list[ModelInterface]

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

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func, domain_name=domain_name)
            ops.add(op)
        for model in self.models:
            ops.add(model.operations(domain_name=domain_name))
        return ops

    def define_operations(self, domain_name: str | None = None) -> Operations:
        ops_col = self.operations(domain_name)
        op_build = StepBuilder()

        with op_build as ob:
            ob.step(ops_col["step_one"])
            ob.step(ops_col["step_two"])
            ob.step(ops_col["step_three"])
            ob.step(ops_col["step_four"])

        return op_build.operations

    def main_loop(self, ctx: Context):
        ops = self.define_operations()
        ops.run(ctx)


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

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func, domain_name=domain_name)
            ops.add(op)
        return ops

    def run(self, ctx: Context): ...


def test_simulation_step_order():
    model1 = MyModel(param=2.0, name="my_model1")
    sim = Simulation(
        domains=[
            Domain(name="region1", solver=MySolver(param1=1.0, models=[model1])),
        ],
        coupling_interface=[],
    )

    ops = sim.domains[0].operations()
    nodes = sim.domains[0].dependencies()
    nodes_ordered = compute_nodes_order(nodes)
    steps_ordered = compute_steps_order(ops)
    assert nodes_ordered == [
        "region1.step_one",
        "region1.initialize",
        "region1.step_two",
        "region1.step_three",
        "region1.process",
        "region1.step_four",
    ]
    steps_names = [op.name for op in steps_ordered]
    assert steps_names == [
        "region1.step_one",
        "region1.initialize",
        "region1.step_two",
        "region1.step_three",
        "region1.process",
        "region1.step_four",
    ]

    # assert len(steps_ordered) == 6

    if True:
        dag = sim.dependency_graph()
        parent_dir = Path(__file__).parent
        digraph_to_pyvis_html(
            dag, html_path=str(parent_dir / "dag_sim_step_order.html")
        )


def test_simulation_run():
    model1 = MyModel(param=2.0, name="my_model1")
    sim = Simulation(
        domains=[
            Domain(name="region1", solver=MySolver(param1=1.0, models=[model1])),
        ],
        coupling_interface=[],
    )

    sim_ctx = sim.init_simulation_context()
    sim_ctx.domain_context["region1"] = sim.domains[0].solver.create_context()
    sim.main_loop(sim_ctx)
    fields = sim_ctx.domain_context["region1"].fields
    assert fields["a"] == 8.0 + 6.0
