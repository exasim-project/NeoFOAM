# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A test-only solver that adds its region's increment of ``cases/regions`` once per step."""

from pathlib import Path
from typing import Any

from neofoam.framework.context import Context
from neofoam.framework.operations import StepBuilder
from neofoam.framework.solver import Solver
from neofoam.io import YAML, BaseConfig, IOStrategy

CASE = Path(__file__).parent / "cases" / "regions"


@IOStrategy(YAML("constant/increment.yaml"))
class IncrementConfig(BaseConfig):
    increment: float


counter = Solver("counter")


@counter.initializer
def _initialize(self: Any) -> Context:
    config = IncrementConfig.load(case_dir=CASE, region=self.region)
    return Context(fields={"total": 0.0}, models={"increment": config.increment})


@counter.time_step
def _time_step(self: Any, ctx: Context, builder: StepBuilder) -> None:
    builder.step(self.operations["add_increment"])


@counter.finalizer
def _finalize(self: Any, ctx: Context) -> None:
    ctx.models["finalized"] = True


@counter.operation()
def add_increment(self: Any, ctx: Context) -> None:
    ctx.fields["total"] += ctx.models["increment"]
