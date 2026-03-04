# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""3-stage initialization for incompressibleFluidNeon solver."""

from pathlib import Path
from typing import Any, Optional

from neofoam.foam.initialization import create_arg_list, create_runtime
from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInit,
    model as init_model,
)
from neofoam.models.stability_criteria import CFLCondition
from neofoam.solver.incompressibleFluidNeon.models.pressure_velocity.base import (
    PressureVelocityAlgorithm,
)


init = StagedInit("incompressibleFluidNeon")


def create_init(case_dir: Optional[Path] = None) -> StagedInit:
    init._case_dir = case_dir  # type: ignore[attr-defined]
    return init


@init.load
def load_config() -> LoadResult:
    pressure_model = PressureVelocityAlgorithm.detect_and_create()
    cfl_condition = CFLCondition()

    core_models: list[Any] = [pressure_model, cfl_condition]

    return LoadResult(core_models=core_models, optional_models=[])


@init.resolve
def resolve_models(config: ConfigContext) -> None:
    # No optional models to resolve
    pass


@init.build
def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
    pressure_model = core_models[0]
    cfl_condition = next(
        (m for m in core_models if isinstance(m, CFLCondition)),
        None,
    )

    builder = InitializerBuilder()

    builder.extend([create_arg_list(init.argv), create_runtime()])
    builder.add_core_models([("pressure_velocity", pressure_model)])

    if cfl_condition is not None:
        builder.add(init_model("cfl_condition", create=lambda _ctx: cfl_condition))

    return builder.build()
