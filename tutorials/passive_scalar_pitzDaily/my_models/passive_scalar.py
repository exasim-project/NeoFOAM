# SPDX-License-Identifier: GPL-3.0-or-later
# Reference implementation matching doc/tutorials/02-write-a-plugin-model.

from pathlib import Path
from typing import Any

import pybFoam as pyf
from pybFoam import (
    fvm,
    fvScalarMatrix,
    surfaceScalarField,
    volScalarField,
)

from neofoam import FieldUpdates, Model, field

# ``neofoam.foam`` (fvSchemes/fvSolution registration decorators) hasn't
# landed in the current branch yet — commented out so the plugin still
# loads. Re-enable once that subpackage is ported.
# from neofoam.foam import fvSchemes, fvSolution
from neofoam.io import BaseConfig
from neofoam.solver.incompressibleFluid.models import incompressibleFluidModel


class PassiveScalarConfig(BaseConfig):
    D: float = 0.0  # diffusivity, m²/s


passive_scalar = Model("passive_scalar").register_with(incompressibleFluidModel)


# Framework note: in the current branch ``@spec.detect`` is called with
# no arguments (the old ``case_dir`` parameter was removed), and ``load``
# receives ``(case_dir, instance_id)`` instead of ``(case_dir, entry)``.
# ``build`` now receives just the config (no ``self``).
@passive_scalar.detect
def detect() -> bool:
    return Path("constant/scalarProperties").exists()


@passive_scalar.load
def load(_case_dir: Path, _instance_id: str) -> PassiveScalarConfig:
    props = pyf.dictionary.read("constant/scalarProperties")
    return PassiveScalarConfig(D=props.get[float]("D"))


@passive_scalar.build
def build(cfg: PassiveScalarConfig) -> list[Any]:
    def create_s(context: dict[str, Any]) -> volScalarField:
        return volScalarField.read_field(context["mesh"], "s")

    return [field("s", create_s, depends_on=["mesh"])]


@passive_scalar.operation(depends_on=["continuity"])
# @fvSchemes.add(
#     ddt="ddt(s)",
#     div="div(phi,s)",
#     laplacian="laplacian(D,s)",
# )
# @fvSolution.add("s")
def solve_s(
    self: Any,
    s: volScalarField,
    phi: surfaceScalarField,
) -> FieldUpdates:
    cfg: PassiveScalarConfig = self.config
    D = pyf.dimensionedScalar("D", pyf.dimViscosity, cfg.D)
    sEqn = fvScalarMatrix(fvm.ddt(s) + fvm.div(phi, s) - fvm.laplacian(D, s))
    sEqn.solve()
    return FieldUpdates({"s": s})
