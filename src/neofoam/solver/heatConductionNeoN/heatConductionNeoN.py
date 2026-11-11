# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""heatConductionNeoN: transient heat conduction on NeoN, coupled regions included.

The same SolverSpec / StagedInit composition as ``incompressibleFluidNeoN``:
the framework ``solutionLoop`` and ``fieldWriter`` around one ``solve_T`` per
time step. ``multiRun`` couples its regions through the coupled patches of ``T``
(:mod:`.coupled_patches`).
"""

from typing import Any, Optional

from neofoam.framework.context import Context
from neofoam.framework.coupling import Coupling, RegionCoupling
from neofoam.framework.operations import IterativeOp, Operation, Operations, StepBuilder
from neofoam.framework.solver import SolverRuntime, SolverSpec, core_model
from neofoam.framework.types import OperationMetadata
from neofoam.solver.incompressibleFluidNeoN.models.solution_loop import SolutionLoopPredicate
from neofoam.solver.neon_runtime import NeoNControlConfig, ensure_neon_initialized
from neofoam.solver.regions import region_case_dir

from .configs import ControlDictConfig, ThermalPropertiesConfig
from .coupled_patches import coupled_patches
from .create_fields import create_init


class _NeoNSolverSpec(SolverSpec):
    """A solver whose runtimes need NeoN: instantiating one brings NeoN up."""

    def instantiate(
        self,
        argv: Optional[list[Any]] = None,
        region: Optional[str] = None,
        coupling: Optional[RegionCoupling] = None,
    ) -> SolverRuntime:
        # not in the initializer: Kokkos is finalized on the thread that initialized it,
        # and multiRun instantiates every region on the main thread, runs it on its own
        ensure_neon_initialized([str(arg) for arg in argv] if argv else [self.name])
        return super().instantiate(argv=argv, region=region, coupling=coupling)


heatConductionNeoN = _NeoNSolverSpec("heatConductionNeoN")

heatConductionNeoN.config(ControlDictConfig)
heatConductionNeoN.config(NeoNControlConfig)
heatConductionNeoN.config(ThermalPropertiesConfig)


@heatConductionNeoN.initializer
def initialize(self: Any) -> Context:
    """The context on the region's case, then the coupling model, then the coupled patches."""
    init = create_init(self.region, list(self.argv))
    ctx = init.run()
    self.core_models = init.core_models
    ctx.models["coupling"] = Coupling(self.coupling)
    ctx.models["conduction"].patches = coupled_patches("T", ctx, region_case_dir(self.region))
    return ctx


@heatConductionNeoN.execution_graph_step
def execution_graph(
    self: Any, ctx: Context, domain_name: Optional[str] = None
) -> tuple[StepBuilder, Operations]:
    """One time step: set_time_step, increment_time, start_step, solve_T, write_output."""
    _ = domain_name
    loop_ops = Operations(core_model(self.state, "solutionLoop").operations)
    writer_ops = Operations(core_model(self.state, "fieldWriter").operations)
    builder = StepBuilder()
    time_loop_op = Operation(
        func=IterativeOp(SolutionLoopPredicate()),
        metadata=OperationMetadata(op_name="time_loop"),
    )
    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(loop_ops["set_time_step"])
        time_builder.step(loop_ops["increment_time"])
        time_builder.step(self.operations["start_step"])
        time_builder.step(self.operations["solve_T"])
        time_builder.step(writer_ops["write_output"])
    return builder, Operations()


@heatConductionNeoN.operation(first=True)
def start_step(self: Any, ctx: Context) -> None:
    """Make the current T the old time; ``first``: once per time step in a coupling loop."""
    ctx.models["conduction"].start_step()


@heatConductionNeoN.operation(depends_on=["start_step"])
def solve_T(self: Any, ctx: Context) -> None:
    """Solve the temperature at the end of the loop's current time step."""
    ctx.models["conduction"].solve()


def run(argv: Optional[list[str]] = None) -> Context:
    """Run the single-region case in the working directory; return the final Context."""
    ctx = heatConductionNeoN.instantiate(argv=argv or ["heatConductionNeoN"]).run()
    print("End")
    return ctx
