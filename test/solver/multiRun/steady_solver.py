# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""A test-only steady region solver: outer iterations, no time loop, no time step.

``x = gain * x_other + offset``, solved once per outer iteration, with the
``x_other`` its ``@coupled`` patch ``interface`` last received (as ``affine``,
which runs in a time loop). ``constant/<region>/steady.yaml`` sets ``gain``,
``offset`` and ``outerIterations``. The context has no time (``ctx.time`` is
``None``) and no solution loop; every solve is appended to
``ctx.models["history"]`` as ``(outer iteration, x)``.
"""

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from neofoam.framework.context import Context
from neofoam.framework.coupling import Array, Coupling, coupled
from neofoam.framework.operations import IterativeOp, Operation, Operations, StepBuilder
from neofoam.framework.solver import Solver
from neofoam.framework.types import OperationMetadata
from neofoam.io import YAML, BaseConfig, IOStrategy


@IOStrategy(YAML("constant/steady.yaml"))
class _SteadyConfig(BaseConfig):
    gain: float
    offset: float
    outerIterations: int = 3


class SteadyData(BaseModel):
    """What a steady region exchanges over its patch: its ``x``."""

    x: Array


@coupled
class Interface:
    """The region's coupled patch ``interface``: sends ``x``, receives the partner's ``x``."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.field = "x"
        self.patch = "interface"
        self.x_other = np.zeros(1)
        Coupling.of(ctx).add(self)

    @coupled.sends
    def send(self) -> SteadyData:
        return SteadyData(x=self.ctx.fields["x"])

    @coupled.receives
    def receive(self, data: SteadyData) -> None:
        self.x_other = data.x


class _Outer:
    """The outer-iteration loop's test: ``count`` iterations."""

    def __init__(self, ctx: Context, count: int) -> None:
        self.ctx, self.count = ctx, count

    def __call__(self, ctx: Context) -> bool:
        ctx.models["outer"] += 1
        return bool(ctx.models["outer"] <= self.count)


steady = Solver("steady")


@steady.initializer
def _initialize(self: Any) -> Context:
    config = _SteadyConfig.load(case_dir=Path("."), region=self.region)
    ctx = Context(fields={"x": np.zeros(1)}, models={"steady": config, "history": [], "outer": 0})
    ctx.models["coupling"] = Coupling(self.coupling)
    ctx.models["interface"] = Interface(ctx)
    return ctx


@steady.execution_graph_step
def _execution_graph(self: Any, ctx: Context) -> tuple[StepBuilder, Operations]:
    """The outer iterations around solve_x."""
    builder = StepBuilder()
    outer = Operation(
        func=IterativeOp(_Outer(ctx, ctx.models["steady"].outerIterations)),
        metadata=OperationMetadata(op_name="outer_loop"),
    )
    with builder.loop(outer) as outer_builder:
        outer_builder.step(self.operations["solve_x"])
    return builder, Operations()


@steady.operation()
def solve_x(self: Any, ctx: Context) -> None:
    """``x = gain * x_other + offset``."""
    config = ctx.models["steady"]
    x_other = ctx.models["interface"].x_other
    ctx.fields["x"] = config.gain * np.asarray(x_other) + config.offset
    ctx.models["history"].append((ctx.models["outer"], float(ctx.fields["x"][0])))
