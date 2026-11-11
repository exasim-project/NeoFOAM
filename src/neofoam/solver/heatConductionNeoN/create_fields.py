# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Staged initialization of heatConductionNeoN: the time loop, the NeoN runtime, ``T``."""

from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf

from neofoam import neofoam_bindings as nfb
from neofoam.framework.initialization import (
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
    field,
    lazy,
)
from neofoam.framework.initialization import model as init_model
from neofoam.framework.model import ModelRuntime
from neofoam.solver.incompressibleFluidNeoN.models.field_writer import NeoNWriteHook, fieldWriter
from neofoam.solver.incompressibleFluidNeoN.models.solution_loop import (
    neon_loop_backend_steps,
    solutionLoop,
)
from neofoam.solver.neon_runtime import NeoNControlConfig
from neofoam.solver.regions import ControlDictConfig as CaseControlDictConfig
from neofoam.solver.regions import region_case_dir

from .conduction import ConductionProblem
from .configs import ControlDictConfig, ThermalPropertiesConfig


def _time_models(region: Optional[str], case_dir: Path) -> tuple[ModelRuntime, ModelRuntime]:
    """The time loop and writer: on the case's ``controlDict.yaml`` for a region."""
    if region is None:
        return solutionLoop.instantiate(case_dir, "main"), fieldWriter.instantiate(case_dir, "main")
    times = CaseControlDictConfig.load(case_dir=Path("."))
    return (
        ModelRuntime(spec=solutionLoop, name="solutionLoop_main", config=times),
        ModelRuntime(spec=fieldWriter, name="fieldWriter_main", config=times),
    )


def _runtime_steps(argv: list[str], case_dir: Path) -> list[InitStep]:
    executor = NeoNControlConfig.load(case_dir=case_dir).executor

    def create_neon_runtime(ctx: dict[str, Any]) -> Any:
        rt = nfb.create_adapter_run_time(ctx["_foam_time"], executor)
        rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
        nfb.map_solver_settings(rt.fv_solution_dict.subDict("solvers"), "T")
        return rt

    return [
        lazy("_arg_list", lambda _ctx: pyf.argList([*argv, "-case", str(case_dir)])),
        # Foam::Time keeps a raw reference to the argList: both live for the run
        lazy("_foam_time", lambda ctx: pyf.Time(ctx["_arg_list"]), depends_on=["_arg_list"]),
        lazy("_neon_runtime", create_neon_runtime, depends_on=["_foam_time"]),
        init_model("neon_runtime", lambda ctx: ctx["_neon_runtime"], depends_on=["_neon_runtime"]),
        lazy("mesh", lambda ctx: ctx["_neon_runtime"].mesh, depends_on=["_neon_runtime"]),
    ]


def _build_steps(core_models: list[Any], argv: list[str], case_dir: Path) -> list[InitStep]:
    solution_loop_model, field_writer_model, _control, properties = core_models

    def inject_writer_hook(ctx: dict[str, Any]) -> NeoNWriteHook:
        hook = NeoNWriteHook(runtime=ctx["_neon_runtime"], scalar_names=("T",), vector_names=())
        ctx["models.writer"].hook = hook
        return hook

    builder = InitializerBuilder()
    builder.extend(_runtime_steps(argv, case_dir))
    builder.add_core_models(
        [("solution_loop_model", solution_loop_model), ("field_writer_model", field_writer_model)]
    )
    builder.extend(neon_loop_backend_steps())
    builder.add(
        field(
            "T",
            lambda ctx: nfb.read_scalar_volume_field(ctx["_neon_runtime"], "T"),
            depends_on=["_neon_runtime"],
            write=True,
        )
    )
    builder.add(
        init_model(
            "conduction",
            lambda ctx: ConductionProblem(ctx["_neon_runtime"], ctx["fields.T"], properties),
            depends_on=["fields.T"],
        )
    )
    builder.add(
        init_model("writer_hook", inject_writer_hook, depends_on=["models.writer", "_neon_runtime"])
    )
    builder.add(
        init_model(
            "step_reporter",
            lambda ctx: ctx["_foam_time"].printExecutionTime,
            depends_on=["_foam_time"],
        )
    )
    builder.add(
        init_model(
            "solutionLoop", lambda _ctx: solution_loop_model, depends_on=["models.solution_loop"]
        )
    )
    return builder.build()


def create_init(region: Optional[str], argv: list[str]) -> StagedInitRunner:
    """A fresh :class:`StagedInitRunner` of heatConductionNeoN on ``region``'s case.

    A region steps on the times of the case's ``system/controlDict.yaml``, a
    single-region case (``region=None``) on its ``system/controlDict``.

    Example::

        ctx = create_init(None, ["heatConductionNeoN"]).run()
    """
    spec_builder = StagedInitSpec.build("heatConductionNeoN")
    case_dir = region_case_dir(region)

    @spec_builder.load
    def load_config() -> LoadResult:
        core_models: list[Any] = [
            *_time_models(region, case_dir),
            ControlDictConfig.load(case_dir=case_dir, validate=False),
            ThermalPropertiesConfig.load(case_dir=case_dir),
        ]
        return LoadResult(core_models=core_models, optional_models=[])

    @spec_builder.build
    def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        return _build_steps(core_models, argv, case_dir)

    return StagedInitRunner(spec_builder.finalize(), argv=argv, region=region)
