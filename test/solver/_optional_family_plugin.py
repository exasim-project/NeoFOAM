# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The example plugin shown in ``doc/reference/cli.rst``."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec
from neofoam.io import OF, BaseConfig, IOStrategy
from neofoam.solver.incompressibleFluid import incompressibleFluid
from neofoam.viscosity import viscosityModel

# docs-existing-family-start
Model("Carreau").register_with(viscosityModel)
# docs-existing-family-end


# docs-new-family-start
@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class radiationModel(BaseModel):
    """Optional radiation models; zero or more run per case."""

    @classmethod
    def all_specs(cls) -> list[ModelSpec]:
        registry = PluginSystem.get_registered("radiationModel")
        return [entry.get_model_instance(entry) for entry in registry.plugin_registry]

    @classmethod
    def detect_models(cls, case_dir: Optional[Path] = None) -> list[ModelRuntime]:
        return [
            spec.instantiate(case_dir=case_dir or Path("."), instance_id=spec.name)
            for spec in cls.all_specs()
            if spec.run_detect()
        ]


@IOStrategy(OF("constant/radiationProperties"))
class RadiationConfig(BaseConfig):
    """The radiation model selected by the case."""

    radiationModel: str


radiationP1 = Model("radiationP1").register_with(radiationModel)
radiationP1.config(RadiationConfig)


@radiationP1.detect
def radiation_is_enabled() -> bool:
    return Path("constant/radiationProperties").is_file()


@radiationP1.operation(depends_on=["momentum"])
def solve_radiation() -> None:
    with Path("radiation_steps.log").open("a") as log:
        log.write("solved\n")


incompressibleFluid.models(radiationModel)
# docs-new-family-end
