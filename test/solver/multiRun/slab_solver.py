# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""``slab``: a test-only region solver of implicit 1D heat conduction, in numpy."""

from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.algorithms.solution_loop.solution_loop import SolutionLoopPredicate, solutionLoop
from neofoam.framework.context import Context
from neofoam.framework.coupling import Array, Coupling, FaceData, Patch, coupled, coupling
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
    """Heat capacity one, so ``diffusivity`` is the conductivity."""

    cells: int
    length: float
    diffusivity: float
    origin: float = 0.0


class _CoupledConductionConfig(BaseModel):
    type: Literal["coupledConduction", "coupledTemperature", "coupledHeatFlux"]
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
        self.patch = patch
        self.ctx = ctx
        self.cell = 0 if patch == "left" else geometry.cells - 1
        self.kappa = geometry.diffusivity
        self.delta = geometry.length / geometry.cells / 2
        self.ref_value = np.zeros(1)
        self.value_fraction = np.ones(1)
        self.heat_in = 0.0

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


class SlabGeometry:
    """The slab's mesh: cells of width ``length / cells`` from ``origin``, faces of area 1."""

    def __init__(self, config: "_SlabConfig") -> None:
        self.config = config

    def face_centres(self, patch: str) -> Any:
        face = self.config.origin + (0.0 if patch == "left" else self.config.length)
        return np.array([[face]])

    def face_areas(self, patch: str) -> Any:
        return np.ones(1)

    def cell_centres(self) -> Any:
        width = self.config.length / self.config.cells
        return self.config.origin + width * (np.arange(self.config.cells)[:, None] + 0.5)

    def cell_volumes(self) -> Any:
        return np.full(self.config.cells, self.config.length / self.config.cells)


class FixedTemperature(SlabPatch):
    def __init__(self, field: str, patch: str, ctx: Context, value: float) -> None:
        super().__init__(field, patch, ctx)
        self.set_mixed(ref_value=[value], value_fraction=[1.0])


class Conduction(FaceData):
    """What a ``coupledConduction`` face exchanges: its sink temperature and conductance."""

    T_sink: Array
    h: Array


@coupled
class CoupledConduction(SlabPatch):
    """A face conducting heat to its partner; both ways a sink temperature and conductance."""

    def __init__(self, field: str, patch: str, ctx: Context, value: float) -> None:
        super().__init__(field, patch, ctx)
        self.set_mixed(ref_value=[value], value_fraction=[1.0])

    @coupled.sends
    def send(self) -> Conduction:
        h = self.kappa / self.delta
        return Conduction(T_sink=self.wall_T() + self.q_out() / h, h=np.full(1, h))

    @coupled.receives
    def receive(self, data: Conduction) -> None:
        self.set_mixed(ref_value=data.T_sink, value_fraction=data.h / (data.h + self.h_own()))


class WallTemperature(FaceData):
    """What the Neumann side sends."""

    T: Array


class WallHeatFlux(FaceData):
    """What the Dirichlet side sends: the heat flux through the wall into its slab."""

    q: Array


@coupled
class CoupledTemperature(SlabPatch):
    """The Dirichlet side: takes the partner's wall temperature, sends its heat flux."""

    def __init__(self, field: str, patch: str, ctx: Context, value: float) -> None:
        super().__init__(field, patch, ctx)
        self.set_mixed(ref_value=[value], value_fraction=[1.0])

    @coupled.sends
    def send(self) -> WallHeatFlux:
        return WallHeatFlux(q=-self.q_out())

    @coupled.receives
    def receive(self, data: WallTemperature) -> None:
        self.set_mixed(ref_value=data.T, value_fraction=[1.0])


@coupled
class CoupledHeatFlux(SlabPatch):
    """The Neumann side: takes the partner's heat flux, sends its wall temperature."""

    def __init__(self, field: str, patch: str, ctx: Context, value: float) -> None:
        super().__init__(field, patch, ctx)
        self.set_mixed(ref_value=[value], value_fraction=[0.0])

    def wall_T(self) -> Any:
        return self.cell_T() + self.heat_in / self.h_own()

    @coupled.sends
    def send(self) -> WallTemperature:
        return WallTemperature(T=self.wall_T())

    @coupled.receives
    def receive(self, data: WallHeatFlux) -> None:
        self.heat_in = -float(data.q[0])


_COUPLED: dict[str, type[SlabPatch]] = {
    "coupledConduction": CoupledConduction,
    "coupledTemperature": CoupledTemperature,
    "coupledHeatFlux": CoupledHeatFlux,
}


def _patch(name: str, entry: Union[float, _CoupledConductionConfig], ctx: Context) -> SlabPatch:
    if isinstance(entry, _CoupledConductionConfig):
        return _COUPLED[entry.type]("T", name, ctx, entry.value)  # type: ignore[call-arg]
    return FixedTemperature("T", name, ctx, entry)


slab = Solver("slab")


def _create_init(region: Optional[str]) -> StagedInitRunner:
    """The loop on the shared controlDict, the slab and temperature of the region."""
    spec_builder = StagedInitSpec.build("slab")
    case_dir = Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        loop = ModelRuntime(
            spec=solutionLoop,
            name="solutionLoop_main",
            config=_ControlDictConfig.load(case_dir=case_dir),
        )
        geometry = _SlabConfig.load(case_dir=Path(region or "."))
        temperature = _TemperatureConfig.load(case_dir=Path(region or "."))
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

    return StagedInitRunner(spec_builder.finalize(), region=region)


@slab.initializer
def _initialize(self: Any) -> Context:
    init = _create_init(self.region)
    ctx = init.run()
    self.core_models = init.core_models
    temperature = ctx.models["temperature"]
    ctx.models["coupling"] = Coupling(self.coupling, SlabGeometry(ctx.models["slab"]))
    ctx.models["patches"] = [
        _patch("left", temperature.left, ctx),
        _patch("right", temperature.right, ctx),
    ]
    for patch in ctx.models["patches"]:
        if hasattr(type(patch), "__coupled__"):
            coupling(ctx).add(Patch(patch.patch), patch)
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
        time_builder.step(self.operations["store_old_time"])
        time_builder.step(self.operations["solve_T"])
    return builder, Operations()


@slab.operation(first=True)
def store_old_time(self: Any, ctx: Context) -> None:
    """Keep the old time's temperature; ``first``: once per time step in a coupling loop."""
    ctx.fields["T_old"] = ctx.fields["T"].copy()


@slab.operation(depends_on=["store_old_time"])
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
        rhs[patch.cell] += h * patch.ref_value[0] + patch.heat_in
    ctx.fields["T"] = np.linalg.solve(matrix, rhs)
