# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""A test-only region solver: explicit 1D heat conduction in a slab, in numpy."""

from pathlib import Path
from typing import Any, Optional

import numpy as np

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.algorithms.solution_loop.solution_loop import SolutionLoopPredicate, solutionLoop
from neofoam.framework.context import Context
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import (
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
    field,
)
from neofoam.framework.initialization import model as init_model
from neofoam.framework.model import ModelRuntime
from neofoam.framework.operations import IterativeOp, Operation, Operations, StepBuilder
from neofoam.framework.solver import Solver
from neofoam.framework.types import OperationMetadata
from neofoam.io import YAML, BaseConfig, IOStrategy


@IOStrategy(YAML("system/controlDict.yaml"))
class _ControlDictConfig(TimeControlConfig):
    pass


@IOStrategy(YAML("constant/slab.yaml"))
class _SlabConfig(BaseConfig):
    cells: int
    length: float
    diffusivity: float


@IOStrategy(YAML("0/T.yaml"))
class _TemperatureConfig(BaseConfig):
    internalField: float
    left: float
    right: float


slab = Solver("slab")


def _create_init(region: Optional[str]) -> StagedInitRunner:
    """The loop on the shared controlDict, the slab and its temperature from the region."""
    spec_builder = StagedInitSpec.build("slab")
    case_dir = Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        loop = ModelRuntime(
            spec=solutionLoop,
            name="solutionLoop_main",
            config=_ControlDictConfig.load(case_dir=case_dir),
        )
        geometry = _SlabConfig.load(case_dir=case_dir, region=region)
        temperature = _TemperatureConfig.load(case_dir=case_dir, region=region)
        return LoadResult(core_models=[loop, geometry, temperature], optional_models=[])

    @spec_builder.build
    def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        loop, geometry, temperature = core_models
        builder = InitializerBuilder()
        builder.add_core_models([("solution_loop_model", loop)])
        builder.add(
            init_model("solutionLoop", lambda _ctx: loop, depends_on=["models.solution_loop"])
        )
        builder.add(init_model("slab", lambda _ctx: geometry))
        builder.add(init_model("temperature", lambda _ctx: temperature))
        builder.add(field("T", lambda _ctx: np.full(geometry.cells, temperature.internalField)))
        return builder.build()

    return StagedInitRunner(spec_builder.finalize())


@slab.initializer
def _initialize(self: Any) -> Context:
    init = _create_init(self.region)
    ctx = init.run()
    self.core_models = init.core_models
    return ctx


@slab.execution_graph_step
def _execution_graph(self: Any, ctx: Context) -> tuple[StepBuilder, Operations]:
    loop_ops = Operations(ctx.models["solutionLoop"].operations)
    builder = StepBuilder()
    time_loop_op = Operation(
        func=IterativeOp(SolutionLoopPredicate()),
        metadata=OperationMetadata(op_name="time_loop"),
    )
    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(loop_ops["set_time_step"])
        time_builder.step(loop_ops["increment_time"])
        self.time_step(ctx, time_builder)
    return builder, Operations()


@slab.time_step
def _time_step(self: Any, ctx: Context, builder: StepBuilder) -> None:
    builder.step(self.operations["solve_T"])


@slab.operation()
def solve_T(self: Any, ctx: Context) -> None:
    """One forward-Euler step; ghost cells hold the fixed face temperatures."""
    geometry = ctx.models["slab"]
    boundary = ctx.models["temperature"]
    T = ctx.fields["T"]
    dx = geometry.length / geometry.cells
    padded = np.concatenate(([2 * boundary.left - T[0]], T, [2 * boundary.right - T[-1]]))
    laplacian = (padded[2:] - 2 * T + padded[:-2]) / dx**2
    ctx.fields["T"] = T + geometry.diffusivity * ctx.time.delta_t * laplacian


def run_alone() -> Context:
    """Run the single-region case in the working directory; return the final Context."""
    runtime = slab.instantiate()
    ctx = runtime.initialize()
    builder, model_ops = runtime.execution_graph(ctx=ctx)
    DAGResolver().resolve(builder, model_ops).operations.run(ctx)
    return ctx
