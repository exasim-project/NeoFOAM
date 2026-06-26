# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NOTE: no `from __future__ import annotations` — keep annotations live so the
# contribution's `phi` / `deltaT` / config params resolve by name.

"""courant — the CFL deltaT limit as a timeStepConstraint contribution.

Solver-side (pybFoam): the contribution injects the live ``phi`` surface flux and
computes the classic CFL limit ``deltaT * maxCo / Co``. It lives under the solver,
not the framework, so ``neofoam.algorithms.solution_loop`` stays pure-Python.

Interim gating (until the interface mechanism auto-gates a contribution from its
owning model's config-presence): the contribution is registered but deactivated at
import, so a run without this model stays fixed-step; participation is toggled via
``timeStepConstraint.activate`` / ``deactivate``. (No interim detect here — gating
is deferred to the auto-gating cutover, and the module is not wired into
``models/__init__`` yet.)
"""

from pybFoam import computeCFLNumber, surfaceScalarField
from pydantic import Field

from neofoam.algorithms.solution_loop.interfaces import VGREAT, timeStepConstraint
from neofoam.io import OF, BaseConfig, IOStrategy

from .incompressibleFluidModel import Model, incompressibleFluidModel

SMALL = 1e-15


@IOStrategy(OF("system/controlDict"))
class CourantConfig(BaseConfig):
    """The CFL ceiling read from ``system/controlDict``."""

    maxCo: float = Field(gt=0.0)


courant = Model("courant").register_with(incompressibleFluidModel)
courant.config(CourantConfig)


@timeStepConstraint.contribute
def courant_limit(phi: surfaceScalarField, deltaT: float, cfg: CourantConfig) -> float:
    """Largest deltaT the CFL condition permits on the live ``phi``.

    ``Co`` is the maximum Courant number on ``phi``; below ``SMALL`` the flow is
    quiescent and the rule offers no opinion (``VGREAT``).
    """
    co = computeCFLNumber(phi)[0]
    if co <= SMALL:
        return VGREAT
    return float(deltaT) * cfg.maxCo / co


# Interim explicit gating (see module docstring): registered but inactive so a run
# without this model is fixed-step; the model/test activates it explicitly.
timeStepConstraint.deactivate(courant_limit)
