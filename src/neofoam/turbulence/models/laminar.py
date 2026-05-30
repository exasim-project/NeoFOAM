# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Native momentum-transport model: ``laminar`` (no turbulence).

``laminar`` is a :class:`ModelSpec` + config + **one operation** + a registered
stress computer — no class. It owns the eddy-viscosity field ``nut``: the model
registers/updates it in the Context through :func:`update_nut`, which writes
``fields.nut`` (a constant zero — laminar flow has no turbulence). It also
**registers how its momentum stress is built** — the shared linear
(Boussinesq) :func:`~neofoam.turbulence.stress.linear_viscous_stress`; a RAS/LES
or non-linear model would register a different stress computer and add its own
k/ε/ω transport operations.
"""

from typing import Any

from neofoam.framework.context import FieldUpdates
from neofoam.viscosity.io import dimensioned_viscosity

from ..config import TurbulencePropertiesConfig
from ..momentumTransport import Model, momentumTransportModel, register_momentum_stress
from ..stress import linear_viscous_stress

__all__ = ["laminar"]

laminar = Model("laminar").register_with(momentumTransportModel)
laminar.config(TurbulencePropertiesConfig)

# The model dispatches its stress computer — laminar reuses the shared linear
# (Boussinesq) assembly. A different closure registers a different function here.
register_momentum_stress(laminar, linear_viscous_stress)


@laminar.operation(operation_number="2.0")  # runs before the momentum predictor
def update_nut(self: Any) -> FieldUpdates:
    """Create / update the eddy-viscosity field ``fields.nut`` (zero for laminar)."""
    return FieldUpdates({"nut": dimensioned_viscosity("nut", 0.0)})
