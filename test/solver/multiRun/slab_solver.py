# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""A test-only region solver: implicit 1D heat conduction in a slab, in numpy.

The heat capacity is one, so ``diffusivity`` is also the conductivity. Each face
of the slab (patch ``left`` / ``right``) is a mixed condition: the heat flux into
its cell is ``h_own * value_fraction * (ref_value - T_cell)``, with ``h_own`` the
conductance between the face and the cell centre. A fixed temperature is
``value_fraction = 1``; ``coupledConduction`` couples the face to a patch of
another region (:class:`CoupledConduction`).
"""

from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.algorithms.solution_loop.solution_loop import SolutionLoopPredicate, solutionLoop
from neofoam.framework.context import Context
from neofoam.framework.coupling import CoupledSide, CouplingOp
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
    # position of the left face; places the patches of coupled regions
    origin: float = 0.0


class _CoupledConductionConfig(BaseModel):
    type: Literal["coupledConduction"]
    value: float


@IOStrategy(YAML("0/T.yaml"))
class _TemperatureConfig(BaseConfig):
    internalField: float
    left: Union[float, _CoupledConductionConfig]
    right: Union[float, _CoupledConductionConfig]


class SlabPatch:
    """A face of the slab, as a mixed condition on the field ``field``."""

    def __init__(self, field: str, patch: str, ctx: Context) -> None:
        geometry = ctx.models["slab"]
        self.field = field
        self.ctx = ctx
        self.cell = 0 if patch == "left" else geometry.cells - 1
        self.kappa = geometry.diffusivity
        self.delta = geometry.length / geometry.cells / 2
        face = geometry.origin + (0.0 if patch == "left" else geometry.length)
        self.coordinates = np.array([[face]])
        self.ref_value = np.zeros(1)
        self.value_fraction = np.ones(1)

    def h_own(self) -> float:
        return float(self.kappa / self.delta)

    def cell_T(self) -> Any:
        return self.ctx.fields[self.field][[self.cell]]

    def wall_T(self) -> Any:
        return self.value_fraction * self.ref_value + (1 - self.value_fraction) * self.cell_T()

    def q_out(self) -> Any:
        """Heat flux from the cell through the face, out of the slab."""
        return self.h_own() * (self.cell_T() - self.wall_T())

    def set_mixed(self, ref_value: Any, value_fraction: Any) -> None:
        self.ref_value = np.asarray(ref_value, dtype=float)
        self.value_fraction = np.asarray(value_fraction, dtype=float)


class FixedTemperature(SlabPatch):
    """A face held at ``value``."""

    def __init__(self, field: str, patch: str, ctx: Context, value: float) -> None:
        super().__init__(field, patch, ctx)
        self.set_mixed(ref_value=[value], value_fraction=[1.0])


class CoupledConduction(SlabPatch):
    """A face conducting heat to the patch of another region it is coupled with."""

    def __init__(self, field: str, patch: str, ctx: Context, value: float) -> None:
        super().__init__(field, patch, ctx)
        self.set_mixed(ref_value=[value], value_fraction=[1.0])
        ctx.graph.wrap(
            ops=ctx.graph.solving(field),
            with_=CouplingOp.shared(ctx, key=(field, patch)),
            side=CoupledSide(send=self.send, receive=self.receive, coordinates=self.coordinates),
        )

    def send(self) -> dict[str, Any]:
        h = self.kappa / self.delta
        return {"T_sink": self.wall_T() + self.q_out() / h, "h": np.full(1, h)}

    def receive(self, T_sink: Any, h: Any) -> None:
        self.set_mixed(ref_value=T_sink, value_fraction=h / (h + self.h_own()))


def _patch(name: str, entry: Union[float, _CoupledConductionConfig], ctx: Context) -> SlabPatch:
    if isinstance(entry, _CoupledConductionConfig):
        return CoupledConduction("T", name, ctx, entry.value)
    return FixedTemperature("T", name, ctx, entry)


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
    """The context, then its patches; a coupled patch registers on the runtime's graph edits."""
    init = _create_init(self.region)
    ctx = init.run()
    self.core_models = init.core_models
    ctx.graph = self.graph
    temperature = ctx.models["temperature"]
    ctx.models["patches"] = [
        _patch("left", temperature.left, ctx),
        _patch("right", temperature.right, ctx),
    ]
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
    builder.step(self.operations["store_old_time"])
    builder.step(self.operations["solve_T"])


@slab.operation()
def store_old_time(self: Any, ctx: Context) -> None:
    """Keep the temperature of the last time step; a coupling loop solves from it again."""
    ctx.fields["T_old"] = ctx.fields["T"].copy()


@slab.operation(solves=["T"], depends_on=["store_old_time"])
def solve_T(self: Any, ctx: Context) -> None:
    """One backward-Euler step from the old time; each patch adds its conductance."""
    geometry = ctx.models["slab"]
    cells = geometry.cells
    dx = geometry.length / cells
    capacity = dx / ctx.time.delta_t
    conductance = geometry.diffusivity / dx
    neighbours = np.full(cells, 2.0)
    neighbours[[0, -1]] = 1.0
    matrix = np.diag(capacity + conductance * neighbours)
    matrix -= conductance * (np.eye(cells, k=1) + np.eye(cells, k=-1))
    rhs = capacity * ctx.fields["T_old"]
    for patch in ctx.models["patches"]:
        h = patch.h_own() * patch.value_fraction[0]
        matrix[patch.cell, patch.cell] += h
        rhs[patch.cell] += h * patch.ref_value[0]
    ctx.fields["T"] = np.linalg.solve(matrix, rhs)


def run_alone() -> Context:
    """Run the single-region case in the working directory; return the final Context."""
    runtime = slab.instantiate()
    ctx = runtime.initialize()
    builder, model_ops = runtime.execution_graph(ctx=ctx)
    ctx.graph.apply(DAGResolver().resolve(builder, model_ops)).operations.run(ctx)
    return ctx
