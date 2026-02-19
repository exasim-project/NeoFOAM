# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""3-stage initialization for incompressibleVoF solver."""

from pathlib import Path
from typing import Any

import pybFoam.vof as vof

from neofoam.foam.initialization import create_time_mesh
from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInit,
    model as init_model,
)
from neofoam.solver.incompressibleVoF import models as _solver_models  # noqa: F401
from neofoam.solver.incompressibleVoF.models.incompressibleVoFModel import (
    incompressibleVoFModel,
)
from neofoam.solver.incompressibleVoF.models.alpha_advection.alphaAdvectionModel import (
    alpha_advection_model,
)
from neofoam.solver.incompressibleVoF.models.pressure_velocity.base import (
    PressureVelocityAlgorithm,
)


init = StagedInit("incompressibleVoF")


def create_init(case_dir: Path = None) -> StagedInit:
    init._case_dir = case_dir
    return init


@init.load
def load_config() -> LoadResult:
    # VoF always uses PIMPLE; detect (and warn on SIMPLE/PISO)
    pressure_model = PressureVelocityAlgorithm.detect_and_create()

    # Detect optional models registered with incompressibleVoFModel
    optional_models = incompressibleVoFModel.detect_models()
    for optional_model in optional_models:
        optional_model.run_load()

    core_models: list[Any] = [alpha_advection_model, pressure_model]
    return LoadResult(core_models=core_models, optional_models=optional_models)


@init.resolve
def resolve_models(config: ConfigContext) -> None:
    for model in init.optional_models:
        model.run_resolve(config)


@init.build
def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
    alpha_model = core_models[0]
    pressure_model = core_models[1]

    builder = InitializerBuilder()
    builder.extend(create_time_mesh(init.argv))
    builder.add_core_models(
        [("alpha_advection", alpha_model), ("pressure_velocity", pressure_model)]
    )

    # Two-phase turbulence model (TwoPhaseTransportModel wraps rho, U, phi, rhoPhi, mixture)
    def create_turbulence(ctx):
        return vof.TwoPhaseTransportModel(
            ctx["fields.rho"],
            ctx["fields.U"],
            ctx["fields.phi"],
            ctx["fields.rhoPhi"],
            ctx["models.mixture"],
        )

    builder.add(
        init_model(
            "turbulence",
            depends_on=[
                "fields.rho",
                "fields.U",
                "fields.phi",
                "fields.rhoPhi",
                "models.mixture",
            ],
            create=create_turbulence,
        )
    )

    builder.add_optional_models(optional_models)
    builder.add_model("optional_models", optional_models)

    return builder.build()
