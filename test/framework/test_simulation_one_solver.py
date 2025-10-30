from foamadapter.framework.dag import NodeData, build_global_dag
from foamadapter.framework.pyvis_utils import digraph_to_pyvis_html
from foamadapter.framework.simulation import Simulation, Domain
from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.solver import Solver
from pydantic import BaseModel
from typing import Literal

@Solver
class FirstSolver(BaseModel):
    name: Literal["FirstSolver"] = "FirstSolver"

    def create_context(self) -> Context:
        ctx = Context(fields={}, models={})
        ctx.fields["a"] = 0.0
        return ctx

    @Solver.step(step_number=1)
    def init(a: float):
        a = 1
        return FieldUpdates({"a": a})

    @Solver.step(step_number=2)
    def factor2(a: float):
        a = a*2
        return FieldUpdates({"a": a})

    @Solver.step(step_number=3)
    def add5(a: float):
        a = a + 5
        return FieldUpdates({"a": a})

    def dependencies(self, domain_name: str) -> list[NodeData]:
        nodedata = []
        for step_info in self._step_data:
            depends_on = [f"{domain_name}.{dep}" for dep in step_info.depends_on]
            nodedata.append(NodeData(name=f"{domain_name}.{step_info.step_name}", depends_on=depends_on, shape="box"))
        return nodedata
    
    def main_loop(self, ctx: Context):
        for step in self._steps:
            step.run(ctx)

def test_simulation_one_solver():
    sim = Simulation(domains=[
        Domain(name="region1", solver=FirstSolver()),
    ], coupling_interface=[])

    # dag = sim.dependency_graph()

    # digraph_to_pyvis_html(dag, "dag.html")


    sim_ctx = sim.init_simulation_context()
    # ctx2 = sim.domains[0].solver.create_context()
    # sim_ctx.domain_context["region1"] = ctx2
    sim.main_loop(sim_ctx)
    sim_ctx
    sim_ctx
    # sim = Simulation(config)
    # assert "region1" in sim.domains
    # assert "region2" in sim.domains
    # assert sim.domains["region1"].model.__class__.__name__ == "SinglePhasePIMPLE"
    # assert sim.domains["region2"].model.__class__.__name__ == "SolidConduction"

