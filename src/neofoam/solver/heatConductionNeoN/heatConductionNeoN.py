# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""heatConductionNeoN — solver entrypoint: heat conduction on NeoN.

Transient solid heat conduction ``dT/dt = laplacian(alpha, T) + Q/(rho cp)``,
``alpha = k/(rho cp)`` from ``constant/thermalProperties``, implicit with the
case's ``ddtSchemes`` / ``laplacianSchemes`` (Euler, or ``steadyState`` for a
steady solve: one outer iteration per time step), discretised by
:class:`~.conduction.ConductionProblem`. The same SolverSpec / StagedInit
composition as ``incompressibleFluidNeoN``: the framework ``solutionLoop`` (with
the NeoN time sync) and ``fieldWriter`` (with the NeoN write hook) around one
``solve_T`` per time step.

It supports region coupling (``multiRun``): its init adds the coupling model,
then the coupled patches of ``T`` (:mod:`.coupled_patches`); the framework's
time-step op agrees on the time step with the regions it is coupled with. A
region is a case directory of its own (:func:`~.create_fields.region_case_dir`).
"""

from typing import Any, Optional

from neofoam.framework.context import Context
from neofoam.framework.coupling import Coupling, RegionCoupling
from neofoam.framework.model import ModelRuntime
from neofoam.framework.operations import IterativeOp, Operation, Operations, StepBuilder
from neofoam.framework.solver import SolverRuntime, SolverSpec
from neofoam.framework.types import OperationMetadata
from neofoam.solver.incompressibleFluidNeoN.models.solution_loop import SolutionLoopPredicate
from neofoam.solver.neon_runtime import NeoNControlConfig, ensure_neon_initialized

from .configs import ControlDictConfig, ThermalPropertiesConfig
from .coupled_patches import coupled_patches
from .create_fields import create_init, region_case_dir


class _NeoNSolverSpec(SolverSpec):
    """A solver whose runtimes need NeoN: instantiating one brings NeoN up.

    Kokkos is initialized once per process and finalized (at exit) on the thread
    that initialized it. ``multiRun`` instantiates every region on the main thread
    and runs it on a thread of its own, so NeoN comes up here, not in the
    initializer.
    """

    def instantiate(
        self,
        argv: Optional[list[Any]] = None,
        region: Optional[str] = None,
        coupling: Optional[RegionCoupling] = None,
    ) -> SolverRuntime:
        ensure_neon_initialized([str(arg) for arg in argv] if argv else [self.name])
        return super().instantiate(argv=argv, region=region, coupling=coupling)


heatConductionNeoN = _NeoNSolverSpec("heatConductionNeoN")

heatConductionNeoN.config(ControlDictConfig)
heatConductionNeoN.config(NeoNControlConfig)
heatConductionNeoN.config(ThermalPropertiesConfig)


def _core_model(state: Any, spec_name: str) -> Any:
    """Find an instantiated core model by its spec name."""
    return next(
        m for m in state.core_models if isinstance(m, ModelRuntime) and m.spec.name == spec_name
    )


@heatConductionNeoN.initializer
def initialize(self: Any) -> Context:
    """The context on the region's case, then the coupling model, then the coupled patches."""
    init = create_init(self.region)
    init.argv = list(self.argv)
    ctx = init.run()
    self.core_models = init.core_models
    ctx.models["coupling"] = Coupling(self.coupling)
    patches = coupled_patches("T", ctx, region_case_dir(self.region))
    ctx.models["patches"] = patches
    ctx.models["conduction"].patches = patches
    return ctx


@heatConductionNeoN.execution_graph_step
def execution_graph(
    self: Any, ctx: Context, domain_name: Optional[str] = None
) -> tuple[StepBuilder, Operations]:
    """One time step::

    set_time_step     # the controlDict deltaT
    <region>.time_step  # if coupled: the smallest of the coupled regions (framework)
    increment_time
    start_step        # the old time level of T, once per time step
    solve_T           # inside the region's coupling op, if coupled
    write_output
    """
    _ = domain_name
    loop_ops = Operations(_core_model(self.state, "solutionLoop").operations)
    writer_ops = Operations(_core_model(self.state, "fieldWriter").operations)
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
    """Make the current T the old time; a coupling op solves from it again each pass.

    Marked ``first``: inside a coupling loop it runs once per time step.
    """
    ctx.models["conduction"].start_step()


@heatConductionNeoN.operation(depends_on=["start_step"])
def solve_T(self: Any, ctx: Context) -> None:
    """Solve the temperature at the end of the loop's current time step."""
    ctx.models["conduction"].solve()


def run(argv: Optional[list[str]] = None) -> Context:
    """Run the single-region case in the working directory; return the final Context."""
    solver = heatConductionNeoN.instantiate(argv=argv or ["heatConductionNeoN"])
    ctx = solver.initialize()
    solver.resolve_operations(ctx).run(ctx)
    print("End")
    return ctx
