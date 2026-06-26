# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NOTE: no `from __future__ import annotations` — keep annotations live so the
# contribution's `phi` / `deltaT` / config params resolve by name.

"""courant — the CFL deltaT limit as a timeStepConstraint contribution.

Solver-side (pybFoam): the contribution injects the live ``phi`` surface flux and
computes the classic CFL limit ``deltaT * maxCo / Co``. It lives under the solver,
not the framework, so ``neofoam.algorithms.solution_loop`` stays pure-Python.

The model registers with the family (so it is discoverable case-free) and the
contribution is **bound to the model** via ``@courant.contributes(timeStepConstraint)``.
Folding is automatic: the contribution participates iff the ``courant`` model is active
for the case — gating is intrinsic to the bound contributor runtimes, matched to the
owning ``timeStepConstraint`` interface by ``ModelSpec`` identity, not a name lookup in
``ctx.models``. No import-time activation toggle.
"""

import os

from pybFoam import computeCFLNumber, dictionary, surfaceScalarField
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


@courant.detect
def detect_model() -> bool:
    """Active iff ``system/controlDict`` declares a ``maxCo`` entry.

    Read relative to the run's working directory (like the other solver-local
    detects); inactive when the file or the entry is absent.
    """
    if not os.path.isfile("system/controlDict"):
        return False
    return bool(dictionary.read("system/controlDict").found("maxCo"))


@courant.contributes(timeStepConstraint)
def courant_limit(phi: surfaceScalarField, deltaT: float, cfg: CourantConfig) -> float:
    """Largest deltaT the CFL condition permits on the live ``phi``.

    ``Co`` is the maximum Courant number on ``phi``; below ``SMALL`` the flow is
    quiescent and the rule offers no opinion (``VGREAT``).
    """
    co = computeCFLNumber(phi)[0]
    if co <= SMALL:
        return VGREAT
    return float(deltaT) * cfg.maxCo / co
