# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""maxDeltaT — a constant deltaT cap as a timeStepConstraint contribution.

A pure-Python optional model: it owns one ``@IOStrategy`` config
(``system/controlDict`` ``maxDeltaT``) and contributes a single float limit to the
``timeStepConstraint`` gather point. The contribution injects only its config (no
pybFoam field) — the field-free half of the CFL/maxDeltaT migration; only the
activation detect reads the case dictionary.

The model registers with the family (so it is discoverable case-free) and the
contribution is **bound to the model** via ``@maxDeltaT.contributes(timeStepConstraint)``.
Folding is automatic: the contribution participates iff the ``maxDeltaT`` model is active
for the case — gating is intrinsic to the bound contributor runtimes, matched to the
owning ``timeStepConstraint`` interface by ``ModelSpec`` identity, not a name lookup in
``ctx.models``. No import-time activation toggle.
"""

import os

from pybFoam import dictionary
from pydantic import Field

from neofoam.algorithms.solution_loop.interfaces import timeStepConstraint
from neofoam.io import OF, BaseConfig, IOStrategy

from .incompressibleFluidModel import Model, incompressibleFluidModel


@IOStrategy(OF("system/controlDict"))
class MaxDeltaTConfig(BaseConfig):
    """The constant deltaT cap read from ``system/controlDict``."""

    maxDeltaT: float = Field(gt=0.0)


maxDeltaT = Model("maxDeltaT").register_with(incompressibleFluidModel)
maxDeltaT.config(MaxDeltaTConfig)


@maxDeltaT.detect
def detect_model() -> bool:
    """Active iff ``system/controlDict`` enables adaptive stepping with a ``maxDeltaT``.

    OpenFOAM only honours ``maxDeltaT`` when ``adjustTimeStep`` is on; mirror that so a
    fixed-step case (``adjustTimeStep no``) keeps a fixed step. Read relative to the
    run's working directory; inactive when the file or the entries are absent.
    """
    if not os.path.isfile("system/controlDict"):
        return False
    cd = dictionary.read("system/controlDict")
    adaptive = cd.found("adjustTimeStep") and cd.get[bool]("adjustTimeStep")
    return bool(adaptive and cd.found("maxDeltaT"))


@maxDeltaT.contributes(timeStepConstraint)
def max_delta_t_limit(cfg: MaxDeltaTConfig) -> float:
    """The largest deltaT this model permits — the constant cap (config only)."""
    return cfg.maxDeltaT
