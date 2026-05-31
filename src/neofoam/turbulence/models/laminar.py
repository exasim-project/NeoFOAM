# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Native momentum-transport model: ``laminar`` (no turbulence).

``laminar`` is a :class:`ModelSpec` + config — no class. Laminar flow has no eddy
viscosity, so the model registers **no ``nut`` field** and solves **no transport
equation** (it contributes no per-step operation). What it owns is **which stress
the momentum equation uses**: its ``@build`` registers the ``viscousStress`` object
— the shared linear eddy-viscosity assembly :class:`LinearViscousStress`
(``nuEff = nu + nut``; an absent ``nut`` means ``nuEff = nu``). The momentum
predictor refreshes that stress (``viscousStress.update``) right where it consumes
it, so there is no separate update operation here.

The assembly is a reusable free class so a RAS/LES closure (kEpsilon, kOmegaSST,
…) reuses it — those models additionally register ``nut`` and a turbulence-transport
``correct`` operation, and a non-linear closure registers a different stress. The
choice always stays with the model.
"""

from typing import Any

from neofoam.framework.initialization import model

from ..config import TurbulencePropertiesConfig
from ..momentumTransport import Model, momentumTransportModel
from ..stress import LinearViscousStress

__all__ = ["laminar"]

laminar = Model("laminar").register_with(momentumTransportModel)
laminar.config(TurbulencePropertiesConfig)


@laminar.build
def build(config: TurbulencePropertiesConfig) -> list[Any]:
    """Inject the ``viscousStress`` the momentum equation uses.

    laminar dispatches the linear eddy-viscosity assembly; the solver runs this
    through ``ModelRuntime.run_build()`` and registers it at ``models.viscousStress``,
    so ``divDevReff`` (and the ``nuEff`` refresh the predictor triggers) is the
    model's decision, not the solver's.
    """

    def create_viscous_stress(_ctx: dict[str, Any]) -> Any:
        return LinearViscousStress()

    return [model("viscousStress", create_viscous_stress)]
