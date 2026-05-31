# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""3-stage initialization for the incompressibleFluid solver (minimal port).

LOAD    → detect the pressure-velocity algorithm and any optional models
RESOLVE → wire optional-model dependencies via ConfigContext
BUILD   → emit lazy InitSteps for mesh, runtime, fields, and transport/turbulence
"""

from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf
from pybFoam.turbulence import (
    incompressibleTurbulenceModel,
    singlePhaseTransportModel,
)

from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
    lazy,
    model as init_model,
)
from .configs import ControlDictConfig
from .models.field_writer import fieldWriter, writer_backend_steps
from .models.incompressibleFluidModel import incompressibleFluidModel
from .models.pressure_velocity.base import PressureVelocityAlgorithm
from .models.solution_loop import loop_backend_steps, solutionLoop


def create_init(case_dir: Optional[Path] = None) -> StagedInitRunner:
    """Build a fresh :class:`StagedInitRunner` for incompressibleFluid.

    The closures over ``runner`` give the resolve/build stages access to
    the live ``argv`` and ``optional_models`` populated by load.
    """
    spec_builder = StagedInitSpec.build("incompressibleFluid")
    resolved_case_dir = case_dir or Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        pressure_model = PressureVelocityAlgorithm.detect_and_create()
        optional_models = incompressibleFluidModel.detect_models(resolved_case_dir)

        # solutionLoop (advances time) and fieldWriter (persists fields) are
        # separate-concern Models, loaded here so their controlDict is validated
        # up front; both are composed by incompressibleFluid.execution_graph.
        solution_loop_model = solutionLoop.instantiate(resolved_case_dir, "main")
        field_writer_model = fieldWriter.instantiate(resolved_case_dir, "main")

        return LoadResult(
            core_models=[pressure_model, solution_loop_model, field_writer_model],
            optional_models=optional_models,
        )

    @spec_builder.resolve
    def resolve_models(config: ConfigContext) -> None:
        for opt in runner.optional_models:
            opt.run_resolve(config)

    @spec_builder.build
    def build_lazy(
        core_models: list[Any], optional_models: list[Any]
    ) -> list[InitStep]:
        pressure_model = core_models[0]

        def _by_spec(spec_name: str) -> Any:
            return next(
                m
                for m in core_models
                if getattr(getattr(m, "spec", None), "name", None) == spec_name
            )

        solution_loop_model = _by_spec("solutionLoop")
        field_writer_model = _by_spec("fieldWriter")
        argv = runner.argv

        def create_foam_time(_ctx: dict[str, Any]) -> Any:
            argList = pyf.argList(argv)
            return pyf.Time(argList)

        def create_mesh(ctx: dict[str, Any]) -> Any:
            return pyf.fvMesh(ctx["_foam_time"])

        def create_laminar_transport(ctx: dict[str, Any]) -> Any:
            return singlePhaseTransportModel(ctx["fields.U"], ctx["fields.phi"])

        def create_turbulence(ctx: dict[str, Any]) -> Any:
            return incompressibleTurbulenceModel.New(
                ctx["fields.U"],
                ctx["fields.phi"],
                ctx["models.laminarTransport"],
            )

        builder = InitializerBuilder()
        # the pybFoam Foam::Time is an init-only resource ("_foam_time"): it
        # parents the mesh objectRegistry and is the write(True) target, but is
        # never routed onto the Context (the leading underscore keeps it hidden).
        builder.add(lazy("_foam_time", create_foam_time))
        builder.add(lazy("mesh", create_mesh, depends_on=["_foam_time"]))

        # solutionLoop + fieldWriter are the framework *core* Models, instantiated
        # as real ModelRuntimes: add_core_models registers them and runs each
        # @build, emitting the LoopState (ctx.time) + engine and the FieldWriter
        # steps. They are backend-agnostic; the pybFoam touch-points (the FoamTime
        # LoopBackend, Courant provider, logger, write hook, step reporter) are
        # injected by the *_backend_steps below through the framework seams.
        builder.add_core_models(
            [
                ("solution_loop_model", solution_loop_model),
                ("field_writer_model", field_writer_model),
            ]
        )
        builder.extend(loop_backend_steps())
        builder.extend(
            writer_backend_steps(ControlDictConfig.load(case_dir=resolved_case_dir))
        )

        # PIMPLE is passed to add_core_models so it lands in models.pressure_velocity.
        # Its lazy field/model InitSteps come from pimple._build_func directly —
        # the ModelSpec is used here as both spec and "runtime" (no instantiate).
        builder.add_core_models([("pressure_velocity", pressure_model)])
        if pressure_model._build_func is not None:
            builder.extend(pressure_model._build_func(pressure_model))

        builder.add(
            init_model(
                "laminarTransport",
                create_laminar_transport,
                depends_on=["fields.U", "fields.phi"],
            )
        )
        builder.add(
            init_model(
                "turbulence",
                create_turbulence,
                depends_on=[
                    "fields.U",
                    "fields.phi",
                    "models.laminarTransport",
                ],
            )
        )

        builder.add_optional_models(optional_models)
        builder.add_model("optional_models", optional_models)

        return builder.build()

    runner = StagedInitRunner(spec_builder.finalize())
    return runner
