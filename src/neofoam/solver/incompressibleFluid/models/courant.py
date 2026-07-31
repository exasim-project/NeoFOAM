# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NOTE: no `from __future__ import annotations` — keep annotations live so the
# contribution's `phi` / `deltaT` / config params resolve by name.

"""courant — the CFL deltaT limit as a timeStepConstraint contribution.

Solver-side (pybFoam): the contributions inject the live ``phi`` surface flux and
compute the classic CFL limit ``deltaT * maxCo / Co`` — one for ``setDeltaT.H``'s
per-step limit, one for ``setInitialDeltaT.H``'s undamped first-step limit. They
live under the solver, not the framework, so ``neofoam.algorithms.solution_loop``
stays pure-Python.

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

from neofoam.algorithms.solution_loop.interfaces import (
    VGREAT,
    initialTimeStepConstraint,
    timeStepConstraint,
)
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
    """Active iff ``system/controlDict`` enables adaptive stepping with a ``maxCo``.

    OpenFOAM only honours ``maxCo`` when ``adjustTimeStep`` is on; mirror that so a
    fixed-step case (``adjustTimeStep no``) keeps a fixed step. Read relative to the
    run's working directory; inactive when the file or the entries are absent.
    """
    if not os.path.isfile("system/controlDict"):
        return False
    cd = dictionary.read("system/controlDict")
    adaptive = cd.found("adjustTimeStep") and cd.get[bool]("adjustTimeStep")
    return bool(adaptive and cd.found("maxCo"))


@courant.contributes(timeStepConstraint)
def courant_limit(phi: surfaceScalarField, deltaT: float, cfg: CourantConfig) -> float:
    """Largest deltaT the CFL condition permits on the live ``phi``.

    Reports the raw maximum, ``maxCo / (Co + SMALL) * deltaT`` — *how* to approach
    that limit (OpenFOAM's ``setDeltaT.H`` growth damping and the 1.2 cap) is the
    solution loop's responsibility (``SolutionLoop.constrain_delta_t``).

    ``Co`` is the maximum Courant number on ``phi``. ``SMALL`` is a denominator
    epsilon, never a cut-off: ``setDeltaT.H`` lets a quiescent start (``Co == 0``)
    produce a huge factor, which the loop's growth cap turns into the usual 20 %
    increase. Treating it as "no opinion" instead would freeze the first step and
    shift the whole trajectory.
    """
    co = computeCFLNumber(phi)[0]
    return cfg.maxCo / (co + SMALL) * float(deltaT)


@courant.contributes(initialTimeStepConstraint)
def courant_initial_limit(phi: surfaceScalarField, deltaT: float, cfg: CourantConfig) -> float:
    """The undamped first-step CFL limit of ``setInitialDeltaT.H``.

    That file is gated on ``CoNum > SMALL``, so a quiescent flow yields no opinion
    (``VGREAT``) and the initial pass — including its write-time snapping — is
    skipped entirely, as in OpenFOAM.
    """
    co = computeCFLNumber(phi)[0]
    if co <= SMALL:
        return VGREAT
    return cfg.maxCo * float(deltaT) / co
