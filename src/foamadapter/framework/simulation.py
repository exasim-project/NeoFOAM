# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from typing import Any

from pydantic import BaseModel

from foamadapter.framework.operations import Operations

from .context import Context
from .couplingInterface import CouplingInterface
from .dag import build_global_dag
from .domain import Domain


class SimulationContext(BaseModel):
    domain_context: dict[str, Context]


class Simulation(BaseModel):
    domains: list[Domain]
    coupling_interface: list[CouplingInterface]

    def dependency_graph(self) -> Any:
        graph = {domain.name: domain.dependencies() for domain in self.domains}
        return build_global_dag(graph)

    def init_simulation_context(self) -> SimulationContext:
        domain_context = {
            domain.name: Context(fields={}, models={}) for domain in self.domains
        }
        # Note: create_context is not part of SolverInterface protocol
        # This would need to be handled differently
        return SimulationContext(domain_context=domain_context)

    def run(self) -> None:
        sim_ctx = self.init_simulation_context()
        self.main_loop(sim_ctx)

    def main_loop(self, sim_ctx: SimulationContext) -> None:
        if len(self.domains) != 1:
            raise NotImplementedError(
                "Only single-domain simulations are supported currently."
            )

        for domain in self.domains:
            ops_collection = domain.execution_graph()
            ctx = sim_ctx.domain_context[domain.name]
            ops = Operations(operations=list(ops_collection.ops))
            ops.run(ctx)
