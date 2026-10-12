# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""multiRun — several solvers, one per region, in one case and one time loop.

The multi-region layout of OpenFOAM.org's ``foamMultiRun``: ``system/controlDict.yaml``
lists the region solvers (:class:`~neofoam.solver.multiRun.configs.ControlDictConfig`)
and each region reads its own files from its region directories. Every region is
an ordinary solver instantiated with ``region=``; it shares this solver's time and
its time-step operations run in this solver's time loop, named ``<region>.<op>``.
Coupled patches of two regions share one :class:`CouplingOp`
(:func:`region_couplings`), so their regions' solutions join one coupling loop.
"""

from typing import Any

from neofoam.algorithms.solution_loop.solution_loop import SolutionLoopPredicate
from neofoam.framework.context import Context
from neofoam.framework.coupling import CouplingOp
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import IterativeOp, Operation, Operations, StepBuilder
from neofoam.framework.solver import Solver, region_operations
from neofoam.framework.solver.registry import resolve_solver
from neofoam.framework.types import OperationMetadata

from .configs import ControlDictConfig
from .create_fields import create_init

multiRun = Solver("multiRun")

multiRun.config(ControlDictConfig)


@multiRun.initializer
def initialize(self: Any) -> Context:
    """Initialize the time loop, then every region on the loop's time.

    Each region starts with its couplings (:func:`region_couplings`); its graph
    edits move to this solver's graph. The regions are kept in
    ``ctx.models["regions"]`` as ``{region: (runtime, ctx)}``.
    """
    init = create_init()
    ctx = init.run()
    self.core_models = init.core_models
    _loop, config = init.core_models
    couplings = region_couplings(list(config.regionSolvers))
    regions: dict[str, tuple[Any, Context]] = {}
    for region, solver_name in config.regionSolvers.items():
        runtime = resolve_solver(solver_name).instantiate(region=region)
        runtime.graph.couplings = couplings.get(region, {})
        region_ctx = runtime.initialize()
        region_ctx.time = ctx.time
        ctx.graph.include_region(region_ctx.graph, region)
        regions[region] = (runtime, region_ctx)
    ctx.models["regions"] = regions
    return ctx


def region_couplings(regions: list[str]) -> dict[str, dict[str, CouplingOp]]:
    """The coupling of each coupled patch, by region; patches sharing one are coupled.

    No interface is known yet, so every region couples its patches by itself.
    """
    return {}


@multiRun.execution_graph_step
def execution_graph(self: Any, ctx: Context) -> tuple[StepBuilder, Operations]:
    """Time loop: set step → advance → each region's time-step operations, in stepping order."""
    loop_ops = Operations(ctx.models["solutionLoop"].operations)

    builder = StepBuilder()
    time_loop_op = Operation(
        func=IterativeOp(SolutionLoopPredicate()),
        metadata=OperationMetadata(op_name="time_loop"),
    )
    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(loop_ops["set_time_step"])
        time_builder.step(loop_ops["increment_time"])
        for runtime, region_ctx in ctx.models["regions"].values():
            for operation in region_operations(runtime, region_ctx):
                time_builder.step(operation)
    return builder, Operations()


@multiRun.finalizer
def finalize(self: Any, ctx: Context) -> None:
    """Finalize every region."""
    for runtime, region_ctx in ctx.models["regions"].values():
        runtime.finalize(region_ctx)


def run() -> Context:
    """Run the multi-region case in the working directory; return the final Context.

    Each region's context is ``ctx.models["regions"][region][1]``.

    Example::

        cd <case>
        python -m neofoam.solver.multiRun
    """
    solver = multiRun.instantiate()
    ctx = solver.initialize()
    builder, model_ops = solver.execution_graph(ctx=ctx)
    ctx.graph.apply(DAGResolver().resolve(builder, model_ops)).operations.run(ctx)
    solver.finalize(ctx)
    return ctx
