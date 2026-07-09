# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NOTE: no `from __future__ import annotations` — keep annotations live so the
# contribution's `phi` / `deltaT` / config params resolve by name.

"""courant — the CFL deltaT limit as a timeStepConstraint contribution (NeoN).

NeoN flavor of the pybFoam solver's ``courant`` model: the contribution
injects the live NeoN ``phi`` surface field and computes the classic CFL
limit ``deltaT * maxCo / Co`` from ``nn.compute_co_num``. Together with the
``NeoNTimeSync`` loop backend this replaces the legacy ``nfb.sync_run_times``
CFL-adjust half (the ``SolutionLoop`` growth cap of 1.2 matches OpenFOAM's
``setDeltaT.H``).

The model registers with the NeoN family (so it is discoverable case-free) and
the contribution is bound to the model via
``@courant.contributes(timeStepConstraint)`` — it participates iff the model is
active for the case (``adjustTimeStep`` + ``maxCo`` in ``controlDict``).
"""

import os
from typing import Any

import neon._neon as nn  # NeoN Python bindings
from pybFoam import dictionary
from pydantic import Field

from neofoam.algorithms.solution_loop.interfaces import VGREAT, timeStepConstraint
from neofoam.io import OF, BaseConfig, IOStrategy

from .incompressibleFluidNeoNModel import Model, incompressibleFluidNeoNModel

SMALL = 1e-15


@IOStrategy(OF("system/controlDict"))
class CourantConfig(BaseConfig):
    """The CFL ceiling read from ``system/controlDict``."""

    maxCo: float = Field(gt=0.0)


courant = Model("courant").register_with(incompressibleFluidNeoNModel)
courant.config(CourantConfig)


@courant.detect
def detect_model() -> bool:
    """Active iff ``system/controlDict`` enables adaptive stepping with a ``maxCo``.

    OpenFOAM only honours ``maxCo`` when ``adjustTimeStep`` is on; mirror that so a
    fixed-step case (``adjustTimeStep no``) keeps a fixed step.
    """
    if not os.path.isfile("system/controlDict"):
        return False
    cd = dictionary.read("system/controlDict")
    adaptive = cd.found("adjustTimeStep") and cd.get[bool]("adjustTimeStep")
    return bool(adaptive and cd.found("maxCo"))


@courant.contributes(timeStepConstraint)
def courant_limit(phi: Any, deltaT: float, cfg: CourantConfig) -> float:
    """Largest deltaT the CFL condition permits on the live NeoN ``phi``.

    ``Co`` is the maximum Courant number on ``phi``; below ``SMALL`` the flow is
    quiescent and the rule offers no opinion (``VGREAT``).
    """
    max_co, _mean_co = nn.compute_co_num(phi, float(deltaT))
    co = float(max_co)
    if co <= SMALL:
        return VGREAT
    return float(deltaT) * cfg.maxCo / co
