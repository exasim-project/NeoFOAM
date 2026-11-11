# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``
# and the dependency resolver matches ``param.annotation is Context``.

"""A test-only region solver on OpenFOAM alone: explicit diffusion, ``T += D laplacian(T)``.

No NeoN, so a region of it runs on part of the MPI processes (``-parallel
-world <region>``). The region named by ``FOAM_DIFFUSION_FAIL`` raises in its
second time step.
"""

import os
from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf
from pybFoam import fvc

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.algorithms.solution_loop.solution_loop import SolutionLoopPredicate, solutionLoop
from neofoam.framework.context import Context
from neofoam.framework.coupling import Coupling
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
from neofoam.io import YAML, IOStrategy

#: The diffusivity times the time step, in m^2: stable on the test meshes (cells >= 0.2 m).
DIFFUSIVITY = 0.01

FAIL_VARIABLE = "FOAM_DIFFUSION_FAIL"


@IOStrategy(YAML("system/controlDict.yaml"))
class _ControlDictConfig(TimeControlConfig):
    pass


class FoamRegion:
    """What a region's OpenFOAM keeps alive for the run: its argList, Time and mesh."""

    def __init__(self, argv: list[str], case_dir: Path) -> None:
        self.arg_list = pyf.argList([*argv, "-case", str(case_dir)])
        self.time = pyf.Time(self.arg_list)
        self.mesh = pyf.fvMesh(self.time)


foamDiffusion = Solver("foamDiffusion")


def _create_init(region: Optional[str], argv: list[str]) -> StagedInitRunner:
    """The loop on the case's controlDict.yaml, OpenFOAM on ``<region>/``, and ``T``."""
    spec_builder = StagedInitSpec.build("foamDiffusion")
    case_dir = Path(region) if region is not None else Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        loop = ModelRuntime(
            spec=solutionLoop,
            name="solutionLoop_main",
            config=_ControlDictConfig.load(case_dir=Path(".")),
        )
        return LoadResult(core_models=[loop], optional_models=[])

    @spec_builder.build
    def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        (loop,) = core_models
        builder = InitializerBuilder()
        builder.add_core_models([("solution_loop_model", loop)])
        builder.add(
            init_model("solutionLoop", lambda _ctx: loop, depends_on=["models.solution_loop"])
        )
        builder.add(init_model("foam", lambda _ctx: FoamRegion(argv, case_dir)))
        builder.add(
            field(
                "T",
                lambda ctx: pyf.volScalarField.read_field(ctx["models.foam"].mesh, "T"),
                depends_on=["models.foam"],
            )
        )
        return builder.build()

    return StagedInitRunner(spec_builder.finalize(), region=region)


@foamDiffusion.initializer
def _initialize(self: Any) -> Context:
    init = _create_init(self.region, list(self.argv))
    ctx = init.run()
    self.core_models = init.core_models
    ctx.models["coupling"] = Coupling(self.coupling)
    return ctx


@foamDiffusion.execution_graph_step
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
        time_builder.step(self.operations["diffuse"])
    return builder, Operations()


@foamDiffusion.operation()
def diffuse(self: Any, ctx: Context) -> None:
    """One explicit step ``T += D * laplacian(T)``; the boundary correction syncs the processors."""
    if ctx.region is not None and os.environ.get(FAIL_VARIABLE) == ctx.region:
        if ctx.time.index == 2:
            raise RuntimeError(f"foamDiffusion: region {ctx.region} fails as asked")
    T = ctx.fields["T"]
    T.assign(T + pyf.dimensionedScalar("D", pyf.dimArea, DIFFUSIVITY) * fvc.laplacian(T))
    T.correctBoundaryConditions()
