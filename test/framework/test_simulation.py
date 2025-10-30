from foamadapter.framework.dag import NodeData, build_global_dag
from foamadapter.framework.pyvis_utils import digraph_to_pyvis_html
from foamadapter.framework.simulation import Simulation, Domain
from foamadapter.framework.solver import Solver
from foamadapter.framework.context import Context
from pydantic import BaseModel
from typing import Literal

@Solver
class FirstSolver(BaseModel):
    name: Literal["FirstSolver"] = "FirstSolver"

    @Solver.step(step_number=1) # check if the steps are sorted
    def step_one(self):
        pass

    @Solver.step(step_number=2)
    def step_two(self):
        pass

    @Solver.step(step_number=3)
    def step_three(self):
        pass

    @Solver.step(step_number=4)
    def step_four(self):
        pass

    def dependencies(self, domain_name: str) -> list[NodeData]:
        nodedata = []
        for step in self._step_data:
            depends_on = [f"{domain_name}.{dep}" for dep in step.depends_on]
            nodedata.append(NodeData(name=f"{domain_name}.{step.step_name}", depends_on=depends_on, shape="box"))
        return nodedata
    
    def main_loop(self, ctx: Context):
        ...

@Solver
class SecondSolver(BaseModel):
    name: Literal["SecondSolver"] = "SecondSolver"

    @Solver.step(step_number=1) # check if the steps are sorted
    def step_one(self):
        pass

    @Solver.step(step_number=2)
    def step_two(self):
        pass

    @Solver.step(step_number=3)
    def step_three(self):
        pass

    @Solver.step(step_number=4)
    def step_four(self):
        pass

    def dependencies(self, domain_name: str) -> list[NodeData]:
        nodedata = []
        for step in self._step_data:
            depends_on = [f"{domain_name}.{dep}" for dep in step.depends_on]
            nodedata.append(NodeData(name=f"{domain_name}.{step.step_name}", depends_on=depends_on, shape="box"))
        return nodedata
    
    def main_loop(self, ctx: Context):
        ...

def test_simulation_initialization():
    sim = Simulation(domains=[
        Domain(name="region1", solver=FirstSolver()),
        Domain(name="region2", solver=FirstSolver()),
        Domain(name="region3", solver=SecondSolver()),
    ], coupling_interface=[])

    dag = sim.dependency_graph()

    digraph_to_pyvis_html(dag, "dag.html")


    # sim = Simulation(config)
    # assert "region1" in sim.domains
    # assert "region2" in sim.domains
    # assert sim.domains["region1"].model.__class__.__name__ == "SinglePhasePIMPLE"
    # assert sim.domains["region2"].model.__class__.__name__ == "SolidConduction"

