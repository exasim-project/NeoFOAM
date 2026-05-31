# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Native viscosity model: ``Newtonian`` (constant viscosity).

``Newtonian`` is a :class:`ModelSpec` + config — no class. It **owns the molecular
viscosity field ``nu``** and registers it itself: its ``@build`` emits the
``fields.nu`` InitStep, built from the ``nu`` value OpenFOAM reads into
``constant/transportProperties``. Newtonian viscosity is constant, so the model
contributes **no correct operation**; a rate-dependent model (CrossPowerLaw,
BirdCarreau, …) is the same shape plus a ``correct`` operation that recomputes
``nu`` from the strain rate after the pressure-velocity coupling.
"""

from typing import Any

from neofoam.framework.initialization import field

from ..config import TransportPropertiesConfig
from ..io import dimensioned_viscosity
from ..viscosityModel import Model, viscosityModel

__all__ = ["newtonian"]

newtonian = Model("Newtonian").register_with(viscosityModel)
newtonian.config(TransportPropertiesConfig)


@newtonian.build
def build(config: TransportPropertiesConfig) -> list[Any]:
    """Register the molecular viscosity ``nu`` the model owns.

    The value comes from the model's own ``transportProperties`` config; the
    solver runs this through ``ModelRuntime.run_build()`` and merges the step.
    """

    def create_nu(_ctx: dict[str, Any]) -> Any:
        return dimensioned_viscosity("nu", config.nu if config.nu is not None else 0.0)

    return [field("nu", create_nu)]
