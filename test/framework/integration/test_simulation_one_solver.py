from foamadapter.framework.dag import NodeData, StepNumber, build_global_dag
from foamadapter.framework.iteration import Iteration
from foamadapter.framework.pyvis_utils import digraph_to_pyvis_html
from foamadapter.framework.simulation import Simulation, Domain
from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.solver import Solver, Step
from pydantic import BaseModel
from typing import Literal


@Iteration
class SubSolver(BaseModel):
    name: Literal["SubSolver"] = "SubSolver"
    max_iterations: int
    current_iteration: int = 0

    def loop(self) -> bool:
        self.current_iteration += 1
        return self.current_iteration <= self.max_iterations

    @Iteration.step(step_number=1)
    def add1(self, a: float):
        a = a + 1
        return FieldUpdates({"a": a})

    @Iteration.step(step_number=2, depends_on=["add1"])
    def add2(self, a: float):
        a = a + 2
        return FieldUpdates({"a": a})

    def run(self, ctx: Context):
        while self.loop():
            for step in self._steps:
                step.run(ctx)

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


@Solver
class FirstSolver(BaseModel):
    name: Literal["FirstSolver"] = "FirstSolver"
    sub_solver: SubSolver

    def create_context(self) -> Context:
        ctx = Context(fields={}, models={})
        ctx.fields["a"] = 0.0
        return ctx

    @Solver.step(step_number=1)
    def init(self, a: float):
        a = 1
        return FieldUpdates({"a": a})

    @Solver.step(step_number=2, depends_on=["init"])
    def factor2(self, a: float):
        a = a * 2
        return FieldUpdates({"a": a})

    @Solver.step(step_number=3, depends_on=["factor2"])
    def sub_iter(self, ctx: Context):
        self.sub_solver.run(ctx)

    @Solver.step(step_number=4, depends_on=["sub_iter"])
    def add5(self, a: float):
        a = a + 5
        return FieldUpdates({"a": a})

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

    def main_loop(self, ctx: Context):
        for step in self.steps():
            step.run(ctx)


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

    digraph_to_pyvis_html(dag, "dag_one_solver.html")


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
    sim.main_loop(sim_ctx)
    ctx_region1 = sim_ctx.domain_context["region1"]
    assert ctx_region1.fields["a"] == 1.0 * 2.0 + 2 * (1.0 + 2.0) + 5.0  # init * factor2 + sub_iter + add5
