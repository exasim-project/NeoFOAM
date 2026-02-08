from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.model import Model
from foamadapter.framework.operations import Operation, OperationCollection, Operations
from foamadapter.framework.pyvis_utils import digraph_to_pyvis_html
from foamadapter.framework.simulation import Domain, Simulation
from foamadapter.framework.solver import Solver


@Model
class SubSolver(BaseModel):
    name: Literal["SubSolver"] = "SubSolver"
    max_iterations: int
    current_iteration: int = 0

    def loop(self) -> bool:
        self.current_iteration += 1
        return self.current_iteration <= self.max_iterations

    @Model.operation(operation_number=1)
    def add1(self, a: float) -> FieldUpdates:
        a = a + 1
        return FieldUpdates({"a": a})

    @Model.operation(operation_number=2, depends_on=["add1"])
    def add2(self, a: float) -> FieldUpdates:
        a = a + 2
        return FieldUpdates({"a": a})

    def run(self, ctx: Context):
        ops = Operations(self.operations())
        while self.loop():
            ops.run(ctx)

    def operations(self, domain_name: Optional[str] = None) -> OperationCollection:
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops


@Solver
class FirstSolver(BaseModel):
    name: Literal["FirstSolver"] = "FirstSolver"
    sub_solver: SubSolver

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
    def sub_iter(self, ctx: Context):
        self.sub_solver.run(ctx)

    @Solver.operation(operation_number=4, depends_on=["sub_iter"])
    def add5(self, a: float):
        a = a + 5
        return FieldUpdates({"a": a})

    def operations(self, domain_name: Optional[str] = None) -> OperationCollection:
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops

    def main_loop(self, ctx: Context):
        ops = Operations(self.operations())
        ops.run(ctx)


def test_sub_solver():
    ctx = Context(fields={}, models={})
    ctx.fields["a"] = 0.0

    subsolver = SubSolver(max_iterations=2)
    subsolver.run(ctx)

    assert ctx.fields["a"] == 6.0


def test_simulation_dag():
    sim = Simulation(
        domains=[
            Domain(
                name="region1",
                solver=FirstSolver(sub_solver=SubSolver(max_iterations=2)),
            ),
        ],
        coupling_interface=[],
    )

    dag = sim.dependency_graph()

    parent_dir = Path(__file__).parent
    path = str(parent_dir / "dag_one_solver.html")
    digraph_to_pyvis_html(dag, path)


def test_simulation_one_solver():
    sim = Simulation(
        domains=[
            Domain(
                name="region1",
                solver=FirstSolver(sub_solver=SubSolver(max_iterations=2)),
            ),
        ],
        coupling_interface=[],
    )

    sim_ctx = sim.init_simulation_context()
    sim_ctx.domain_context["region1"] = sim.domains[0].solver.create_context()
    sim.main_loop(sim_ctx)
    ctx_region1 = sim_ctx.domain_context["region1"]
    assert (
        ctx_region1.fields["a"] == 1.0 * 2.0 + 2 * (1.0 + 2.0) + 5.0
    )  # init * factor2 + sub_iter + add5
