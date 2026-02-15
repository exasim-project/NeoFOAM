# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""3-stage initialization for incompressibleFluid solver."""

from pathlib import Path
from typing import Any

from neofoam.foam.initialization import create_time_mesh
from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInit,
    model as init_model,
)
from neofoam.models.stability_criteria import CFLCondition
from neofoam.transportModels.transport_model import TransportModel
from neofoam.turbulenceModels.turbulence import TurbulenceModel
from neofoam.solver.incompressibleFluid import models as _solver_models  # noqa: F401
from neofoam.solver.incompressibleFluid.models.incompressibleFluidModel import (
    incompressibleFluidModel,
)
from neofoam.solver.incompressibleFluid.models.pressure_velocity.base import (
    PressureVelocityAlgorithm,
)


init = StagedInit("incompressibleFluid")


def create_init(case_dir: Path = None) -> StagedInit:
    init._case_dir = case_dir
    return init


@init.load
def load_config() -> LoadResult:
    # Directly detect and create pressure-velocity algorithm (not a plugin)
    pressure_model = PressureVelocityAlgorithm.detect_and_create()

    cfl_condition = CFLCondition()

    # Detect optional models (e.g., boussinesq)
    optional_models = incompressibleFluidModel.detect_models()
    for optional_model in optional_models:
        optional_model.run_load()

    return LoadResult(
        core_models=[pressure_model, cfl_condition],
        optional_models=optional_models,
    )


@init.resolve
def resolve_models(config: ConfigContext) -> None:
    # Resolve optional models (e.g., boussinesq)
    # They can access the pressure model via init.core_models[0]
    for model in init.optional_models:
        model.run_resolve(config)


@init.build
def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
    # The first core model is the pressure-velocity algorithm (pimple/simple/piso)
    pressure_model = core_models[0]
    cfl_condition = next(
        model for model in core_models if isinstance(model, CFLCondition)
    )

    builder = InitializerBuilder()
    builder.extend(create_time_mesh(init.argv))
    builder.add_core_models([("pressure_velocity", pressure_model)])

    builder.add(
        init_model(
            "laminarTransport",
            depends_on=["fields.U", "fields.phi"],
            create=lambda ctx: TransportModel.from_type("singlePhase").create_instance(
                ctx["fields.U"], ctx["fields.phi"]
            ),
        )
    )

    builder.add(
        init_model(
            "turbulence",
            depends_on=["fields.U", "fields.phi", "models.laminarTransport"],
            create=lambda ctx: TurbulenceModel.from_type(
                "openfoam_rts"
            ).create_instance(
                ctx["fields.U"],
                ctx["fields.phi"],
                ctx["models.laminarTransport"],
            ),
        )
    )

    builder.add_optional_models(optional_models)
    builder.add_model("optional_models", optional_models)

    return builder.build()
