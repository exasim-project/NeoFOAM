# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Top-level Simulation orchestrator: domains, coupling, and main loop."""

from typing import Any

from pydantic import BaseModel

from neofoam.framework.graph import build_global_dag

from .context import Context
from .couplingInterface import CouplingInterface
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
            ops = domain.operations()
            ctx = sim_ctx.domain_context[domain.name]
            ops.run(ctx)
