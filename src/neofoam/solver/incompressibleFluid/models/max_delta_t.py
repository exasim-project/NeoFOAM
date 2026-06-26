# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""maxDeltaT — a constant deltaT cap as a timeStepConstraint contribution.

A pure-Python optional model: it owns one ``@IOStrategy`` config
(``system/controlDict`` ``maxDeltaT``) and contributes a single float limit to the
``timeStepConstraint`` gather point. The contribution injects only its config (no
pybFoam field), so it is the backend-free half of the CFL/maxDeltaT migration.

Interim gating (until the interface mechanism auto-gates a contribution from its
owning model's config-presence):

* the **model** is active iff ``maxDeltaT`` is present in ``system/controlDict`` — a
  config-presence ``@detect`` that lazily reads the dict and degrades to inactive when
  pybFoam, the file, or the entry is absent. (The spec's no-detect final form lands
  with the auto-gating cutover; without a detect this always-active model would pollute
  every solver run's optional-model detection.)
* the **contribution** is registered but **deactivated at import** so a run without
  this model stays fixed-step; participation is toggled explicitly via
  ``timeStepConstraint.activate`` / ``deactivate``.
"""

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
    """Active iff ``system/controlDict`` declares a ``maxDeltaT`` entry.

    Reads the dict lazily via pybFoam relative to the run's working directory (like the
    other solver-local detects); returns ``False`` when pybFoam, the file, or the entry
    is absent.
    """
    try:
        import pybFoam as pyf

        control_dict = pyf.dictionary.read("system/controlDict")
        return bool(control_dict.found("maxDeltaT"))
    except (ImportError, RuntimeError):
        return False


@timeStepConstraint.contribute
def max_delta_t_limit(cfg: MaxDeltaTConfig) -> float:
    """The largest deltaT this model permits — the constant cap (config only)."""
    return cfg.maxDeltaT


# Interim explicit gating (see module docstring): registered but inactive so a run
# without this model is fixed-step; the model/test activates it explicitly.
timeStepConstraint.deactivate(max_delta_t_limit)
