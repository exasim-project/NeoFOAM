# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Native momentum-transport model: ``laminar`` (no turbulence).

``laminar`` is a :class:`ModelSpec` + config — no class. Laminar flow has no eddy
viscosity, so the model registers **no ``nut`` field** and contributes **no
correct operation**. What it *does* define is **which stress the momentum equation
uses**: its ``@build`` registers the ``viscousStress`` model — here the linear
(Boussinesq) eddy-viscosity assembly ``LinearViscousStress`` (``nuEff = nu + nut``;
an absent ``nut`` means ``nuEff = nu``). A RAS/LES closure is the same shape but
additionally registers ``nut`` and a k/ε/ω ``correct`` operation, and a non-linear
closure registers a different stress here — the choice stays with the model.
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
    """Register the ``viscousStress`` that computes the momentum stress term.

    laminar dispatches the linear eddy-viscosity assembly; the solver runs this
    through ``ModelRuntime.run_build()`` and merges the step, so ``divDevReff`` is
    the model's decision, not the solver's.
    """

    def create_viscous_stress(_ctx: dict[str, Any]) -> Any:
        return LinearViscousStress()

    return [model("viscousStress", create_viscous_stress)]
