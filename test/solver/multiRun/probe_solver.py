# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""``probe``: a test-only solver of one number, ``offset + sum(gain * received)``."""

from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, create_model

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.algorithms.solution_loop.solution_loop import SolutionLoopPredicate, solutionLoop
from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Array,
    Coupling,
    FaceData,
    Global,
    Patch,
    RegionCoupling,
    Scalar,
    ValueData,
    coupled,
    coupling,
)
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
    """``gains`` keyed ``<port>.<name>``; each port is a ``@coupled`` patch, or a Global one.

    ``values``: the Global ports, each with the data field it receives, one value.
    """

    field: Literal["x", "y"] = "x"
    offset: float = 0.0
    gains: dict[str, float] = {}
    coordinates: bool = True
    values: dict[str, str] = {}
    deltaT: Optional[float] = None


class X(FaceData):
    """What a region that solves ``x`` exchanges."""

    x: Array


class Y(FaceData):
    """What a region that solves ``y`` exchanges."""

    y: Array


EXCHANGE: dict[str, Union[type[X], type[Y]]] = {"x": X, "y": Y}


class ProbeGeometry:
    """Every patch one face at 0, or none without ``coordinates``."""

    def __init__(self, coordinates: bool) -> None:
        self.coordinates = coordinates

    def face_centres(self, patch: str) -> Any:
        return np.zeros((1, 1)) if self.coordinates else None

    def face_areas(self, patch: str) -> Any:
        return np.ones(1)

    def cell_centres(self) -> Any:
        return np.zeros((1, 1))

    def cell_volumes(self) -> Any:
        return np.ones(1)


class ProbePatch:
    """A coupled patch of the region: keeps what arrives as ``<patch>.<name>``."""

    def __init__(self, ctx: Context, field: str, patch: str) -> None:
        self.ctx, self.field, self.patch = ctx, field, patch

    def keep(self, data: BaseModel) -> None:
        for name, value in data:
            self.ctx.models["received"][f"{self.patch}.{name}"] = float(np.ravel(value)[0])


def patch_class(field: str, received: str) -> type[ProbePatch]:
    """The ``@coupled`` patch of a region solving ``field`` that receives ``received``."""
    Sent, Received = EXCHANGE[field], EXCHANGE[received]

    @coupled
    class _Patch(ProbePatch):
        @coupled.sends
        def send(self) -> Sent:  # type: ignore[valid-type]
            return Sent.model_validate({field: self.ctx.fields[field]})

        @coupled.receives
        def receive(self, data: Received) -> None:  # type: ignore[valid-type]
            self.keep(data)

    _Patch.__qualname__ = f"ProbePatch[{Sent.__name__} -> {Received.__name__}]"
    return _Patch


def _value_class(name: str) -> type[ProbePatch]:
    """The ``@coupled`` Global port that receives one value of the data field ``name``."""
    received = create_model(  # type: ignore[call-overload]
        f"Value_{name}", __base__=ValueData, **{name: (Scalar, ...)}
    )

    def receive(self: ProbePatch, data: Any) -> None:
        self.keep(data)

    receive.__annotations__ = {"data": received, "return": None}
    return coupled(type("_Value", (ProbePatch,), {"receive": coupled.receives(receive)}))


def add_patches(
    ctx: Context, config: ProbeConfig, region_coupling: Optional[RegionCoupling]
) -> None:
    """The region's field, its coupling model and a coupled patch per patch of ``gains``."""
    ctx.fields[config.field] = np.zeros(1)
    ctx.models.update(
        probe=config,
        received={},
        history=[],
        coupling=Coupling(region_coupling, ProbeGeometry(config.coordinates)),
    )
    ctx.models["patches"] = []
    received: dict[str, str] = {}
    for port, name in config.values.items():
        instance = _value_class(name)(ctx, config.field, port)
        coupling(ctx).add(Global(port), instance)
        ctx.models["patches"].append(instance)
    for key in config.gains:
        patch, name = key.split(".")
        if patch in config.values:
            continue
        if received.setdefault(patch, name) != name:
            raise ValueError(f"patch {patch} receives one field, got {received[patch]} and {name}")
    for patch, name in sorted(received.items()):
        instance = patch_class(config.field, name)(ctx, config.field, patch)
        coupling(ctx).add(Patch(patch), instance)
        ctx.models["patches"].append(instance)


def solve(ctx: Context, index: int) -> None:
    """``field = offset + sum(gain * received)``, recorded as ``(index, value)``."""
    config = ctx.models["probe"]
    received = ctx.models["received"]
    value = config.offset + sum(g * received.get(key, 0.0) for key, g in config.gains.items())
    ctx.fields[config.field] = np.full(1, value)
    ctx.models["history"].append((index, value))


probe = Solver("probe")


def _create_init(region: Optional[str]) -> StagedInitRunner:
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

    return StagedInitRunner(spec.finalize(), region=region)


@probe.initializer
def _initialize(self: Any) -> Context:
    init = _create_init(self.region)
    ctx = init.run()
    self.core_models = init.core_models
    config = ProbeConfig.load(case_dir=Path(self.region or "."))
    add_patches(ctx, config, self.coupling)
    if config.deltaT is not None:
        ctx.models["solution_loop"].set_delta_t(config.deltaT, adjust=False)
    return ctx


@probe.execution_graph_step
def _execution_graph(self: Any, ctx: Context) -> tuple[StepBuilder, Operations]:
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


@probe.operation()
def solve_x(self: Any, ctx: Context) -> None:
    solve(ctx, ctx.time.index)


@probe.operation()
def solve_y(self: Any, ctx: Context) -> None:
    solve(ctx, ctx.time.index)
