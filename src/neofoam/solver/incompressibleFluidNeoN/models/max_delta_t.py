# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""maxDeltaT — a constant deltaT cap as a timeStepConstraint contribution.

NeoN-family twin of the pybFoam solver's ``maxDeltaT`` model: config-only (no
backend field is injected), so the body is identical — only the family it
registers with differs.
"""

import os

from pybFoam import dictionary
from pydantic import Field

from neofoam.algorithms.solution_loop.interfaces import timeStepConstraint
from neofoam.io import OF, BaseConfig, IOStrategy

from .incompressibleFluidNeoNModel import Model, incompressibleFluidNeoNModel


@IOStrategy(OF("system/controlDict"))
class MaxDeltaTConfig(BaseConfig):
    """The constant deltaT cap read from ``system/controlDict``."""

    maxDeltaT: float = Field(gt=0.0)


maxDeltaT = Model("maxDeltaT").register_with(incompressibleFluidNeoNModel)
maxDeltaT.config(MaxDeltaTConfig)


@maxDeltaT.detect
def detect_model() -> bool:
    """Active iff ``system/controlDict`` enables adaptive stepping with a ``maxDeltaT``."""
    if not os.path.isfile("system/controlDict"):
        return False
    cd = dictionary.read("system/controlDict")
    adaptive = cd.found("adjustTimeStep") and cd.get[bool]("adjustTimeStep")
    return bool(adaptive and cd.found("maxDeltaT"))


@maxDeltaT.contributes(timeStepConstraint)
def max_delta_t_limit(cfg: MaxDeltaTConfig) -> float:
    """The largest deltaT this model permits — the constant cap (config only)."""
    return cfg.maxDeltaT
