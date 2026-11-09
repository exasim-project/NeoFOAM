# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""A test-only region solver that does arithmetic on one number: ``x = gain * x_other + offset``.

It exists to test the multi-region framework, not physics. Each region reads
``constant/<region>/affine.yaml`` (``gain``, ``offset``, ``initial``, and
optionally ``deltaT``, the time step the region asks for); its one
operation ``solve_x`` sets the region's field ``x`` from
``x_other``, the value the region last received over its patch ``interface``.
The patch is ``@coupled`` and exchanges :class:`AffineData` (``x``) both ways:
it sends its ``x`` and keeps the partner's as ``x_other``, so two regions
coupled at ``interface`` iterate ``x_a = gain_a * x_b + offset_a``,
``x_b = gain_b * x_a + offset_b`` to their fixed point. Every solve is appended
to ``ctx.models["history"]`` as ``(time index, x)``, which shows the passes of the
coupling loop.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.algorithms.solution_loop.solution_loop import SolutionLoopPredicate, solutionLoop
from neofoam.framework.context import Context
from neofoam.framework.coupling import Array, Coupling, coupled
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

# Every region sits at the same single point of the interface: nearest-neighbour
# mapping (the only mapping regionCoupling.yaml offers) then is the identity.
POINT = np.zeros((1, 1))


@IOStrategy(YAML("system/controlDict.yaml"))
class _ControlDictConfig(TimeControlConfig):
    pass


@IOStrategy(YAML("constant/affine.yaml"))
class _AffineConfig(BaseConfig):
    gain: float
    offset: float
    # x before the first solve, and x_other before the first receive
    initial: float = 0.0
    # the time step the region asks for, instead of the controlDict's deltaT
    deltaT: Optional[float] = None


class AffineData(BaseModel):
    """What an affine region exchanges over its patch: its ``x``."""

    x: Array


@coupled
class Interface:
    """The region's coupled patch ``interface``: sends ``x``, receives the partner's ``x``."""

    def __init__(self, ctx: Context, initial: float) -> None:
        self.ctx = ctx
        self.field = "x"
        self.patch = "interface"
        self.x_other = np.full(1, initial)
        Coupling.of(ctx).add(self)

    @coupled.coordinates
    def points(self) -> Any:
        return POINT

    @coupled.sends
    def send(self) -> AffineData:
        return AffineData(x=self.ctx.fields["x"])

    @coupled.receives
    def receive(self, data: AffineData) -> None:
        self.x_other = data.x


affine = Solver("affine")


def _create_init(region: Optional[str]) -> StagedInitRunner:
    """The loop on the shared controlDict, the region's affine.yaml and its field ``x``."""
    spec_builder = StagedInitSpec.build("affine")
    case_dir = Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        loop = ModelRuntime(
            spec=solutionLoop,
            name="solutionLoop_main",
            config=_ControlDictConfig.load(case_dir=case_dir),
        )
        coefficients = _AffineConfig.load(case_dir=case_dir, region=region)
        return LoadResult(core_models=[loop, coefficients], optional_models=[])

    @spec_builder.build
    def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        loop, coefficients = core_models
        builder = InitializerBuilder()
        builder.add_core_models([("solution_loop_model", loop)])
        builder.add(
            init_model("solutionLoop", lambda _ctx: loop, depends_on=["models.solution_loop"])
        )
        builder.add(init_model("affine", lambda _ctx: coefficients))
        builder.add(field("x", lambda _ctx: np.full(1, coefficients.initial)))
        return builder.build()

    return StagedInitRunner(spec_builder.finalize())


@affine.initializer
def _initialize(self: Any) -> Context:
    """The context, its coupling model, then its coupled patch, which adds itself to it."""
    init = _create_init(self.region)
    ctx = init.run()
    self.core_models = init.core_models
    ctx.models["history"] = []
    ctx.models["coupling"] = Coupling(self.coupling)
    ctx.models["interface"] = Interface(ctx, ctx.models["affine"].initial)
    if ctx.models["affine"].deltaT is not None:
        ctx.models["solution_loop"].set_delta_t(ctx.models["affine"].deltaT, adjust=False)
    return ctx


@affine.execution_graph_step
def _execution_graph(self: Any, ctx: Context) -> tuple[StepBuilder, Operations]:
    """The time loop: set the time step, agree on it, advance the time, solve x."""
    loop_ops = Operations(ctx.models["solutionLoop"].operations)
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(SolutionLoopPredicate()), metadata=OperationMetadata(op_name="time_loop")
    )
    with builder.loop(time_loop) as time_builder:
        time_builder.step(loop_ops["set_time_step"])
        time_builder.step(self.operations["agree_dt"])
        time_builder.step(loop_ops["increment_time"])
        time_builder.step(self.operations["solve_x"])
    return builder, Operations()


@affine.operation()
def solve_x(self: Any, ctx: Context) -> None:
    """``x = gain * x_other + offset``."""
    coefficients = ctx.models["affine"]
    x_other = ctx.models["interface"].x_other
    ctx.fields["x"] = coefficients.gain * x_other + coefficients.offset
    ctx.models["history"].append((ctx.time.index, float(ctx.fields["x"][0])))


@affine.operation()
def agree_dt(self: Any, ctx: Context) -> None:
    """The coupled regions step the smallest time step any of them asks for."""
    loop = ctx.models["solution_loop"]
    loop.set_delta_t(Coupling.of(ctx).agree(loop.current_delta_t()), adjust=False)
