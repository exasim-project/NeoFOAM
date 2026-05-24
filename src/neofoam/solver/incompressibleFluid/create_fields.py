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
from neofoam.models.stability_criteria import CFLCondition

from .configs import ControlDictConfig, TransportPropertiesConfig
from .models.incompressibleFluidModel import incompressibleFluidModel
from .models.pressure_velocity.base import PressureVelocityAlgorithm


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

        # CFLCondition is only meaningful for transient algorithms; SIMPLE
        # is steady-state so we skip it there.
        core_models: list[Any] = [pressure_model]
        if getattr(pressure_model, "algorithm_type", "").upper() != "SIMPLE":
            core_models.append(CFLCondition())

        # Solver-core configs, so ``LoadResult.configs`` exposes the
        # configs the solver consumes (collectible / savable / printable).
        # ``validate=False`` keeps loading lenient — ``LoadResult.validate()``
        # reports any errors. ``pressure_model`` stays at index 0; these
        # append after it and are inert in the BUILD stage (selected there
        # by isinstance).
        #
        # The PIMPLE ``fvSchemes`` / ``fvSolution`` slices are *declared* on
        # the ``pimple`` spec, so ``LoadResult.config_classes`` lists them as
        # part of the schema set. Their instances are not loaded here: the
        # OpenFOAM reader does not yet parse the typed scheme values
        # (``DivScheme`` …) / ``dict`` solver entries those slices carry.
        core_models += [
            ControlDictConfig.load(case_dir=resolved_case_dir, validate=False),
            TransportPropertiesConfig.load(case_dir=resolved_case_dir, validate=False),
        ]

        return LoadResult(
            core_models=core_models,
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
        cfl_condition = next(
            (m for m in core_models if isinstance(m, CFLCondition)), None
        )
        argv = runner.argv

        def create_runtime(_ctx: dict[str, Any]) -> Any:
            argList = pyf.argList(argv)
            return pyf.Time(argList)

        def create_mesh(ctx: dict[str, Any]) -> Any:
            return pyf.fvMesh(ctx["runtime"])

        def create_laminar_transport(ctx: dict[str, Any]) -> Any:
            return singlePhaseTransportModel(ctx["fields.U"], ctx["fields.phi"])

        def create_turbulence(ctx: dict[str, Any]) -> Any:
            return incompressibleTurbulenceModel.New(
                ctx["fields.U"],
                ctx["fields.phi"],
                ctx["models.laminarTransport"],
            )

        builder = InitializerBuilder()
        builder.add(lazy("runtime", create_runtime))
        builder.add(lazy("mesh", create_mesh, depends_on=["runtime"]))

        # PIMPLE is passed to add_core_models so it lands in models.pressure_velocity.
        # Its lazy field/model InitSteps come from pimple._build_func directly —
        # the ModelSpec is used here as both spec and "runtime" (no instantiate).
        builder.add_core_models([("pressure_velocity", pressure_model)])
        if pressure_model._build_func is not None:
            builder.extend(pressure_model._build_func(pressure_model))

        if cfl_condition is not None:
            builder.add(init_model("cfl_condition", lambda _ctx: cfl_condition))

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
