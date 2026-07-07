# SPDX-License-Identifier: GPL-3.0-or-later
# Reference implementation matching doc/tutorials/03-build-a-solver.

from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf
from pybFoam import surfaceScalarField, volScalarField, volVectorField

from neofoam import StagedInit, field
from neofoam.foam.initialization import create_time_mesh
from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
    LoadResult,
)


init = StagedInit("scalar_transport")


def create_init(case_dir: Optional[Path] = None) -> StagedInit:
    init._case_dir = case_dir  # type: ignore[attr-defined]
    return init


@init.load
def load_config() -> LoadResult:
    return LoadResult(core_models=[], optional_models=[])


@init.resolve
def resolve_models(config: ConfigContext) -> None:
    pass


@init.build
def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
    builder = InitializerBuilder()
    builder.extend(create_time_mesh(init.argv))

    def create_T(ctx: dict[str, Any]) -> volScalarField:
        return volScalarField.read_field(ctx["mesh"], "T")

    def create_U(ctx: dict[str, Any]) -> volVectorField:
        return volVectorField.read_field(ctx["mesh"], "U")

    def create_D(ctx: dict[str, Any]) -> volScalarField:
        return volScalarField.read_field(ctx["mesh"], "D")

    def create_phi(ctx: dict[str, Any]) -> surfaceScalarField:
        return pyf.createPhi(ctx["fields.U"])

    builder.add(field("T", create_T, depends_on=["mesh"]))
    builder.add(field("U", create_U, depends_on=["mesh"]))
    builder.add(field("D", create_D, depends_on=["mesh"]))
    builder.add(field("phi", create_phi, depends_on=["fields.U"]))

    return builder.build()
