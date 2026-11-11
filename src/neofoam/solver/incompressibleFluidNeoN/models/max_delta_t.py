# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""maxDeltaT — a constant deltaT ceiling as a maxTimeStep contribution.

NeoN-family twin of the pybFoam solver's ``maxDeltaT`` model: config-only (no
backend field is injected), so the body is identical — only the family it
registers with differs.
"""

from pathlib import Path

from pybFoam import dictionary
from pydantic import Field

from neofoam.algorithms.solution_loop.interfaces import maxTimeStep
from neofoam.io import OF, BaseConfig, IOStrategy

from .incompressibleFluidNeoNModel import Model, incompressibleFluidNeoNModel


@IOStrategy(OF("system/controlDict"))
class MaxDeltaTConfig(BaseConfig):
    """The constant deltaT cap read from ``system/controlDict``."""

    maxDeltaT: float = Field(gt=0.0)


maxDeltaT = (
    Model("maxDeltaT")
    .register_with(incompressibleFluidNeoNModel)
    .labeled("Time step limit (maxDeltaT)")
)
maxDeltaT.config(MaxDeltaTConfig)


@maxDeltaT.detect
def detect_model(case_dir: Path) -> bool:
    """Active iff ``system/controlDict`` enables adaptive stepping with a ``maxDeltaT``."""
    control_dict = case_dir / "system" / "controlDict"
    if not control_dict.is_file():
        return False
    cd = dictionary.read(str(control_dict))
    adaptive = cd.found("adjustTimeStep") and cd.get[bool]("adjustTimeStep")
    return bool(adaptive and cd.found("maxDeltaT"))


@maxDeltaT.contributes(maxTimeStep)
def max_delta_t_limit(cfg: MaxDeltaTConfig) -> float:
    """The constant deltaT ceiling this model imposes (config only).

    A ``maxTimeStep`` ceiling rather than a ``timeStepConstraint`` limit because
    ``setDeltaT.H`` clips it onto the *already-damped* step.
    """
    return cfg.maxDeltaT
