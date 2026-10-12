# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case config of the multiRun solver: the shared time loop and the region solvers."""

from pydantic import Field

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.io import YAML, IOStrategy


@IOStrategy(YAML("system/controlDict.yaml"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict.yaml`` — the time loop every region shares, and its solvers.

    ``regionSolvers`` maps each region to the solver that runs it, in stepping
    order; each region keeps its own files under ``constant/<region>/``,
    ``0/<region>/`` and ``system/<region>/``. A region solver registers its
    time-step body with ``@<solver>.time_step``.

    Example::

        endTime: 10.0
        deltaT: 1.0
        regionSolvers:
          fluid: myFluidSolver
          solid: mySolidSolver
    """

    # The i18n prefix labels the wizard's "add a region" row (neofoam.ui.form_schema).
    regionSolvers: dict[str, str] = Field(min_length=1, json_schema_extra={"i18n": "nf.region"})
