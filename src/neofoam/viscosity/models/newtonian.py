# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Native viscosity model: ``Newtonian`` (constant viscosity).

``Newtonian`` is a :class:`ModelSpec` + config + **one operation** — no class.
It owns the molecular viscosity field ``nu``: the model registers/updates it in
the Context through :func:`update_nu`, which writes ``fields.nu``. Newtonian
viscosity is constant, so the operation simply (re)publishes the value from
``constant/transportProperties``; a rate-dependent model (CrossPowerLaw,
BirdCarreau, …) is the same shape but recomputes ``nu`` from the strain rate.
"""

from typing import Any

from neofoam.framework.context import FieldUpdates

from ..config import TransportPropertiesConfig
from ..io import dimensioned_viscosity
from ..viscosityModel import Model, viscosityModel

__all__ = ["newtonian"]

newtonian = Model("Newtonian").register_with(viscosityModel)
newtonian.config(TransportPropertiesConfig)


@newtonian.operation(operation_number="2.0")  # runs before the momentum predictor
def update_nu(self: Any) -> FieldUpdates:
    """Create / update the molecular viscosity field ``fields.nu``."""
    return FieldUpdates({"nu": dimensioned_viscosity("nu", self.config.nu)})
