# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""controlDict time config + the Foam::Time advancement constants/helpers.

:class:`TimeControlConfig` is the validated ``system/controlDict`` slice the loop
loads. The constants and helpers (``_WRITE_CONTROL_ALIASES``, ``_round_half_away``,
``SMALL``, ``LABEL_MAX``) encode the Foam::Time advancement semantics and are used
by :class:`~neofoam.algorithms.solution_loop.solution_loop.SolutionLoop` when it
advances a :class:`~neofoam.algorithms.solution_loop.loop_state.LoopState`. Mirrors
``$FOAM_SRC/OpenFOAM/db/Time/{Time.C,TimeState.C}`` (v2406).
"""

from __future__ import annotations

import math

from pydantic import Field

from neofoam.algorithms.field_writer.write_control import WriteControlConfig
from neofoam.io import OF, IOStrategy

SMALL = 1e-15
LABEL_MAX = 2**31 - 1

# write-control kinds (canonical OpenFOAM keywords + the "adjustable" alias)
_TIME_STEP = "timeStep"
_RUN_TIME = "runTime"
_ADJUSTABLE = "adjustableRunTime"
_NONE = "none"
_WRITE_CONTROL_ALIASES = {
    "timeStep": _TIME_STEP,
    "runTime": _RUN_TIME,
    "adjustable": _ADJUSTABLE,
    "adjustableRunTime": _ADJUSTABLE,
    "none": _NONE,
}


def _round_half_away(x: float) -> int:
    """Foam::round — round half away from zero (≠ Python banker's round)."""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


@IOStrategy(OF("system/controlDict"))
class TimeControlConfig(WriteControlConfig):
    """Time-stepping + write control, as a validated config.

    Extends :class:`~neofoam.algorithms.field_writer.write_control.WriteControlConfig`
    with the advancement keys the core loop reads (``endTime``/``deltaT``).
    Adaptive stepping (``maxCo``/``maxDeltaT``) is opt-in via the ``courant`` /
    ``max_delta_t`` models, not the core config.
    """

    endTime: float = Field(gt=0)
    deltaT: float = Field(gt=0)
