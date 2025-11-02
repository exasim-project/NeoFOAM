from pydantic import BaseModel

from foamadapter.framework.step import Step
from .solver import SolverInterface
from .couplingInterface import CouplingInterface
from .dag import build_global_dag, NodeData
from .context import Context


class SimulationContext(BaseModel):
    domain_context: dict[str, Context]

class Domain(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    solver: SolverInterface

    def dependencies(self) -> list[NodeData]:
        return self.solver.dependencies(self.name)
    
    def steps(self) -> list[Step]:
        return self.solver.steps(domain_name=self.name)


# def run_solver(step: SolverInterface, ctx: Context):
#     for step in step._steps:
#         step.run(ctx)
            

class Simulation(BaseModel):
    domains: list[Domain]
    coupling_interface: list[CouplingInterface]

    def dependency_graph(self):
        graph = {domain.name: domain.dependencies() for domain in self.domains}
        return build_global_dag(graph)
    
    def init_simulation_context(self) -> SimulationContext:
        domain_context = {domain.name: Context(fields={}, models={}) for domain in self.domains}
        for domain in self.domains:
            domain_ctx = domain.solver.create_context()
            domain_context[domain.name] = domain_ctx
        return SimulationContext(domain_context=domain_context)
    
    def run(self):

        sim_ctx = self.init_simulation_context()
        self.main_loop(sim_ctx)

    def main_loop(self, sim_ctx: SimulationContext):


        if len(self.domains) != 1:
            raise NotImplementedError("Only single-domain simulations are supported currently.")
        
        for domain in self.domains:
            steps = domain.steps()
            ctx = sim_ctx.domain_context[domain.name]
            for step in steps:
                step.run(ctx)
            # domain.solver.main_loop(ctx)
            # run_solver(domain.solver, ctx)
