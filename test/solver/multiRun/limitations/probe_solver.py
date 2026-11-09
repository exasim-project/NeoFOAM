# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""``probe`` — a test-only region solver to probe the coupling design with trivial math.

Each region solves one field, ``x`` or ``y`` (``solve_x`` / ``solve_y``), as::

    field = offset + sum(gain * received)

``constant/<region>/probe.yaml`` sets ``field``, ``offset`` and ``gains``, keyed
``<patch>.<name>``: e.g. ``to_b.y: 0.5`` adds half of the ``y`` that arrives over
patch ``to_b``. Every patch named in ``gains`` is a ``@coupled`` patch: it sends
its field (:class:`X` or :class:`Y`) and receives the ``<name>`` of its gains,
which it keeps.

The knob ``coordinates: false`` gives the patches no points, on purpose. Every
solve appends the new value to ``ctx.models["history"]``.
"""

from pathlib import Path
from typing import Any, Literal, Union

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


@IOStrategy(YAML("constant/probe.yaml"))
class ProbeConfig(BaseConfig):
    field: Literal["x", "y"] = "x"
    offset: float = 0.0
    gains: dict[str, float] = {}
    coordinates: bool = True


class X(BaseModel):
    """What a region that solves ``x`` exchanges: its value."""

    x: Array


class Y(BaseModel):
    """What a region that solves ``y`` exchanges: its value."""

    y: Array


#: The exchange model of each field a region can solve.
EXCHANGE: dict[str, Union[type[X], type[Y]]] = {"x": X, "y": Y}


class ProbePatch:
    """A coupled patch of the region: keeps what arrives as ``<patch>.<name>``.

    :func:`patch_class` makes the ``@coupled`` class for what it sends and receives.
    """

    def __init__(self, ctx: Context, field: str, patch: str, coordinates: bool) -> None:
        self.ctx, self.field, self.patch = ctx, field, patch
        self.has_points = coordinates
        Coupling.of(ctx).add(self)

    def points(self) -> Any:
        return np.zeros((1, 1)) if self.has_points else None

    def keep(self, data: BaseModel) -> None:
        for name, value in data:
            self.ctx.models["received"][f"{self.patch}.{name}"] = float(np.asarray(value)[0])


def patch_class(field: str, received: str) -> type[ProbePatch]:
    """The coupled patch of a region solving ``field`` that receives the partner's ``received``."""
    Sent, Received = EXCHANGE[field], EXCHANGE[received]

    @coupled
    class _Patch(ProbePatch):
        @coupled.coordinates
        def points(self) -> Any:
            return super().points()

        @coupled.sends
        def send(self) -> Sent:  # type: ignore[valid-type]
            return Sent.model_validate({field: self.ctx.fields[field]})

        @coupled.receives
        def receive(self, data: Received) -> None:  # type: ignore[valid-type]
            self.keep(data)

    _Patch.__qualname__ = f"ProbePatch[{Sent.__name__} -> {Received.__name__}]"
    return _Patch


probe = Solver("probe")


def _create_init() -> StagedInitRunner:
    spec = StagedInitSpec.build("probe")

    @spec.load
    def load() -> LoadResult:
        config = _ControlDictConfig.load(case_dir=Path("."))
        loop = ModelRuntime(spec=solutionLoop, name="solutionLoop_main", config=config)
        return LoadResult(core_models=[loop], optional_models=[])

    @spec.build
    def build(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        (loop,) = core_models
        builder = InitializerBuilder()
        builder.add_core_models([("solution_loop_model", loop)])
        builder.add(
            init_model("solutionLoop", lambda _ctx: loop, depends_on=["models.solution_loop"])
        )
        return builder.build()

    return StagedInitRunner(spec.finalize())


@probe.initializer
def _initialize(self: Any) -> Context:
    init = _create_init()
    ctx = init.run()
    self.core_models = init.core_models
    config = ProbeConfig.load(case_dir=Path("."), region=self.region)
    ctx.fields[config.field] = np.zeros(1)
    ctx.models.update(probe=config, received={}, history=[], coupling=Coupling(self.coupling))
    received: dict[str, str] = {}
    for key in config.gains:
        patch, name = key.split(".")
        if received.setdefault(patch, name) != name:
            raise ValueError(f"patch {patch} receives one field, got {received[patch]} and {name}")
    ctx.models["patches"] = [
        patch_class(config.field, name)(ctx, config.field, patch, config.coordinates)
        for patch, name in sorted(received.items())
    ]
    return ctx


@probe.execution_graph_step
def _execution_graph(self: Any, ctx: Context) -> tuple[StepBuilder, Operations]:
    """The time loop: set the time step, advance the time, solve the field.

    A coupled region's time-step op agrees the time step in between.
    """
    loop_ops = Operations(ctx.models["solutionLoop"].operations)
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(SolutionLoopPredicate()), metadata=OperationMetadata(op_name="time_loop")
    )
    with builder.loop(time_loop) as time_builder:
        time_builder.step(loop_ops["set_time_step"])
        time_builder.step(loop_ops["increment_time"])
        time_builder.step(self.operations[f"solve_{ctx.models['probe'].field}"])
    return builder, Operations()


def _solve(ctx: Context) -> None:
    config = ctx.models["probe"]
    received = ctx.models["received"]
    value = config.offset + sum(g * received.get(key, 0.0) for key, g in config.gains.items())
    ctx.fields[config.field] = np.full(1, value)
    ctx.models["history"].append(value)


@probe.operation()
def solve_x(self: Any, ctx: Context) -> None:
    """``x = offset + sum(gain * received)``."""
    _solve(ctx)


@probe.operation()
def solve_y(self: Any, ctx: Context) -> None:
    """``y = offset + sum(gain * received)``."""
    _solve(ctx)
