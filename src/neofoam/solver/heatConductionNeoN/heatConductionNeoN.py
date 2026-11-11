# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""heatConductionNeoN: transient heat conduction on NeoN, coupled regions included.

A patch of ``0/T`` is coupled when its ``coupled`` entry names the condition::

    interface
    {
        type        calculated;
        coupled     coupledTemperature;   // or coupledHeatFlux
        value       uniform 300;
    }
"""

from typing import Any, Optional

from neofoam.framework.context import Context
from neofoam.framework.coupling import Participant, Patch, coupling
from neofoam.framework.operations import IterativeOp, Operation, Operations, StepBuilder
from neofoam.framework.solver import Solver, core_model
from neofoam.framework.types import OperationMetadata
from neofoam.solver.incompressibleFluidNeoN.models.solution_loop import SolutionLoopPredicate
from neofoam.solver.neon_runtime import (
    NeoNControlConfig,
    NeoNFields,
    NeoNGeometry,
    check_written_patches,
    ensure_neon_initialized,
)
from neofoam.solver.regions import region_case_dir

from .configs import ControlDictConfig, ThermalPropertiesConfig
from .coupled_patches import coupled_patches
from .create_fields import create_init

heatConductionNeoN = Solver("heatConductionNeoN")
# Kokkos is finalized on its initializing thread; multiRun instantiates on the main one
heatConductionNeoN.on_instantiate = ensure_neon_initialized

heatConductionNeoN.config(ControlDictConfig)
heatConductionNeoN.config(NeoNControlConfig)
heatConductionNeoN.config(ThermalPropertiesConfig)


@heatConductionNeoN.initializer
def initialize(self: Any) -> Context:
    init = create_init(self.region, list(self.argv))
    ctx = init.run()
    self.core_models = init.core_models
    if self.coupling is not None:
        time_dir = region_case_dir(self.region) / ctx.models["solution_loop"].timeName()
        check_written_patches(self.coupling.field_ports, time_dir)
    runtime = ctx.models["neon_runtime"]
    ctx.models["coupling"] = Participant(
        self.coupling, NeoNGeometry(runtime), NeoNFields(runtime, ctx.fields)
    )
    patches = coupled_patches("T", ctx, region_case_dir(self.region))
    for patch, bc in patches.items():
        coupling(ctx).add(Patch(patch), bc)
    ctx.models["conduction"].patches = list(patches.values())
    return ctx


@heatConductionNeoN.execution_graph_step
def execution_graph(
    self: Any, ctx: Context, domain_name: Optional[str] = None
) -> tuple[StepBuilder, Operations]:
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
