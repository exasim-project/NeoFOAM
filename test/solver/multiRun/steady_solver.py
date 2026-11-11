# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""``steady``: the ``probe`` solver in ``outerIterations`` outer iterations, without a time.

``<region>/constant/steady.yaml`` holds a probe config plus ``outerIterations``;
``ctx.time`` is ``None`` and there is no solution loop. A solve records
``(outer iteration, value)``.
"""

from pathlib import Path
from typing import Any

from neofoam.framework.context import Context
from neofoam.framework.operations import IterativeOp, Operation, Operations, StepBuilder
from neofoam.framework.solver import Solver
from neofoam.framework.types import OperationMetadata
from neofoam.io import YAML, IOStrategy
from solver.multiRun.probe_solver import ProbeConfig, add_patches, solve


@IOStrategy(YAML("constant/steady.yaml"))
class _SteadyConfig(ProbeConfig):
    outerIterations: int = 3


class _Outer:
    """The outer loop's test: ``count`` iterations."""

    def __init__(self, count: int) -> None:
        self.count = count

    def __call__(self, ctx: Context) -> bool:
        ctx.models["outer"] += 1
        return bool(ctx.models["outer"] <= self.count)


steady = Solver("steady")


@steady.initializer
def _initialize(self: Any) -> Context:
    ctx = Context(fields={}, models={"outer": 0})
    add_patches(ctx, _SteadyConfig.load(case_dir=Path(self.region or ".")), self.coupling)
    return ctx


@steady.execution_graph_step
def _execution_graph(self: Any, ctx: Context) -> tuple[StepBuilder, Operations]:
    """The outer iterations around solve_x."""
    builder = StepBuilder()
    outer = Operation(
        func=IterativeOp(_Outer(ctx.models["probe"].outerIterations)),
        metadata=OperationMetadata(op_name="outer_loop"),
    )
    with builder.loop(outer) as outer_builder:
        outer_builder.step(self.operations["solve_x"])
    return builder, Operations()


@steady.operation()
def solve_x(self: Any, ctx: Context) -> None:
    solve(ctx, ctx.models["outer"])
