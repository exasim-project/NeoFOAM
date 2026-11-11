# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""maxDeltaT — a constant deltaT ceiling as a maxTimeStep contribution.

A pure-Python optional model: it owns one ``@IOStrategy`` config
(``system/controlDict`` ``maxDeltaT``) and contributes a single float limit to the
``maxTimeStep`` gather point. The contribution injects only its config (no
pybFoam field) — the field-free half of the CFL/maxDeltaT migration; only the
activation detect reads the case dictionary.

The model registers with the family (so it is discoverable case-free) and the
contribution is **bound to the model** via ``@maxDeltaT.contributes(maxTimeStep)``.
Folding is automatic: the contribution participates iff the ``maxDeltaT`` model is active
for the case — gating is intrinsic to the bound contributor runtimes, matched to the
owning ``maxTimeStep`` interface by ``ModelSpec`` identity, not a name lookup in
``ctx.models``. No import-time activation toggle.
"""

from pathlib import Path

from pybFoam import dictionary
from pydantic import Field

from neofoam.algorithms.solution_loop.interfaces import maxTimeStep
from neofoam.io import OF, BaseConfig, IOStrategy

from .incompressibleFluidModel import Model, incompressibleFluidModel


@IOStrategy(OF("system/controlDict"))
class MaxDeltaTConfig(BaseConfig):
    """The constant deltaT cap read from ``system/controlDict``."""

    maxDeltaT: float = Field(gt=0.0)


maxDeltaT = (
    Model("maxDeltaT")
    .register_with(incompressibleFluidModel)
    .labeled("Time step limit (maxDeltaT)")
)
maxDeltaT.config(MaxDeltaTConfig)


@maxDeltaT.detect
def detect_model(case_dir: Path) -> bool:
    """Active iff ``system/controlDict`` enables adaptive stepping with a ``maxDeltaT``.

    OpenFOAM only honours ``maxDeltaT`` when ``adjustTimeStep`` is on; mirror that so a
    fixed-step case (``adjustTimeStep no``) keeps a fixed step. Read from ``case_dir``;
    inactive when the file or the entries are absent.
    """
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
