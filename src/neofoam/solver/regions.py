# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What the solvers share to run as the coupled regions of a ``multiRun`` case."""

from pydantic import Field

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.io import YAML, IOStrategy

__all__ = ["ControlDictConfig"]


@IOStrategy(YAML("system/controlDict.yaml"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict.yaml``: the time control of every region, and its solvers.

    ``participants`` maps each region to its solver, in stepping order; a region
    reads its own files under ``<region>/``.

    Example::

        endTime: 10.0
        deltaT: 1.0
        participants:
          fluid: myFluidSolver
          solid: mySolidSolver
    """

    # The i18n prefix labels the wizard's "add a region" row (neofoam.ui.form_schema).
    participants: dict[str, str] = Field(min_length=1, json_schema_extra={"i18n": "nf.region"})
