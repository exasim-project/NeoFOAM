# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""3-stage initialization for the heatConductionNeoN solver.

LOAD    → the time loop, the field writer and the thermal properties of the case
BUILD   → lazy InitSteps for the NeoN runtime, the mesh, ``fields.T`` and
          ``models.conduction`` (:class:`~.conduction.ConductionProblem`)

The NeoN ``RunTime`` is built as in ``incompressibleFluidNeoN``: a pybFoam
``Foam::Time`` on the case, the adapter (``create_adapter_run_time``) on it, its
OpenFOAM ``fvSchemes`` / ``fvSolution`` mapped to NeoN once. Its mesh is the
case's default region (``constant/polyMesh``): a region of a multi-region case
is a case directory of its own (see :func:`region_case_dir`), stepping on the
times of the case's ``system/controlDict.yaml``.
"""

from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
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
from neofoam.solver.multiRun.configs import ControlDictConfig as CaseControlDictConfig
from neofoam.solver.neon_runtime import NeoNControlConfig

from .conduction import ConductionProblem
from .configs import ControlDictConfig, ThermalPropertiesConfig


def region_case_dir(region: Optional[str]) -> Path:
    """The case directory of ``region``: ``<region>/`` of the working directory, or ``.``.

    NeoFOAM's runtime reads the mesh of the default region only
    (``constant/polyMesh``), so a region of a multi-region case is an OpenFOAM
    case of its own: ``<region>/{0,constant,system}``, next to the case's
    ``system/controlDict.yaml`` (``regionSolvers``) and ``system/regionCoupling.yaml``.
    """
    return Path(region) if region is not None else Path(".")


def create_init(region: Optional[str] = None) -> StagedInitRunner:
    """Build a fresh :class:`StagedInitRunner` for heatConductionNeoN on ``region``'s case.

    A region of a multi-region case steps on the case's times, as every region of
    a ``multiRun`` case does: its time loop and writer read
    ``system/controlDict.yaml``; a single-region case reads ``system/controlDict``.
    """
    spec_builder = StagedInitSpec.build("heatConductionNeoN")
    resolved_case_dir = region_case_dir(region)

    @spec_builder.load
    def load_config() -> LoadResult:
        if region is None:
            solution_loop_model = solutionLoop.instantiate(resolved_case_dir, "main")
            field_writer_model = fieldWriter.instantiate(resolved_case_dir, "main")
        else:
            times = CaseControlDictConfig.load(case_dir=Path("."))
            solution_loop_model = ModelRuntime(
                spec=solutionLoop, name="solutionLoop_main", config=times
            )
            field_writer_model = ModelRuntime(
                spec=fieldWriter, name="fieldWriter_main", config=times
            )
        core_models: list[Any] = [
            solution_loop_model,
            field_writer_model,
            ControlDictConfig.load(case_dir=resolved_case_dir, validate=False),
            ThermalPropertiesConfig.load(case_dir=resolved_case_dir),
        ]
        return LoadResult(core_models=core_models, optional_models=[])

    @spec_builder.build
    def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        solution_loop_model, field_writer_model, _control, properties = core_models
        argv = [*runner.argv, "-case", str(resolved_case_dir)]
        executor = NeoNControlConfig.load(case_dir=resolved_case_dir).executor

        def create_arg_list(_ctx: dict[str, Any]) -> Any:
            return pyf.argList(argv)

        def create_foam_time(ctx: dict[str, Any]) -> Any:
            # Foam::Time keeps a raw reference to the argList: both stay alive for
            # the run (the NeoNTimeSync backend pins them).
            return pyf.Time(ctx["_arg_list"])

        def create_neon_runtime(ctx: dict[str, Any]) -> Any:
            rt = nfb.create_adapter_run_time(ctx["_foam_time"], executor)
            rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
            nfb.map_solver_settings(rt.fv_solution_dict.subDict("solvers"), "T")
            return rt

        def create_t(ctx: dict[str, Any]) -> Any:
            return nfb.read_scalar_volume_field(ctx["_neon_runtime"], "T")

        def create_conduction(ctx: dict[str, Any]) -> ConductionProblem:
            return ConductionProblem(ctx["_neon_runtime"], ctx["fields.T"], properties)

        def inject_writer_hook(ctx: dict[str, Any]) -> NeoNWriteHook:
            hook = NeoNWriteHook(runtime=ctx["_neon_runtime"], scalar_names=("T",), vector_names=())
            ctx["models.writer"].hook = hook
            return hook

        def make_reporter(ctx: dict[str, Any]) -> Any:
            registry = ctx["_foam_time"]
            return lambda: registry.printExecutionTime()

        builder = InitializerBuilder()
        builder.add(lazy("_arg_list", create_arg_list))
        builder.add(lazy("_foam_time", create_foam_time, depends_on=["_arg_list"]))
        builder.add(lazy("_neon_runtime", create_neon_runtime, depends_on=["_foam_time"]))
        builder.add(
            init_model(
                "neon_runtime", lambda ctx: ctx["_neon_runtime"], depends_on=["_neon_runtime"]
            )
        )
        builder.add(
            lazy("mesh", lambda ctx: ctx["_neon_runtime"].mesh, depends_on=["_neon_runtime"])
        )
        builder.add_core_models(
            [
                ("solution_loop_model", solution_loop_model),
                ("field_writer_model", field_writer_model),
            ]
        )
        builder.extend(neon_loop_backend_steps())
        builder.add(field("T", create_t, depends_on=["_neon_runtime"], write=True))
        builder.add(init_model("conduction", create_conduction, depends_on=["fields.T"]))
        builder.add(
            init_model(
                "writer_hook", inject_writer_hook, depends_on=["models.writer", "_neon_runtime"]
            )
        )
        builder.add(init_model("step_reporter", make_reporter, depends_on=["_foam_time"]))
        builder.add(
            init_model(
                "solutionLoop",
                lambda _ctx: solution_loop_model,
                depends_on=["models.solution_loop"],
            )
        )
        return builder.build()

    runner = StagedInitRunner(spec_builder.finalize(), region=region)
    return runner
