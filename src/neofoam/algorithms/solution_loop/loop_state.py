# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""LoopState — the inspectable solution-loop state (pure data).

The relevant time-stepping information for one run, held as a plain dataclass so
it is trivially inspectable and serialisable. :class:`~neofoam.algorithms.solution_loop.solution_loop.SolutionLoop`
*advances* it (the Foam::Time ``operator++`` arithmetic, write-time decision and
``deltaT`` adjustment all live there); a :class:`LoopBackend` (e.g. the pybFoam
``FoamTime``) optionally mirrors it onto a C++ ``Foam::Time``.

The field names ``value`` / ``index`` / ``write_time`` are exactly the read
surface a :class:`~neofoam.algorithms.field_writer.write_control.StepView`
needs, so a ``LoopState`` satisfies that protocol structurally.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoopState:
    """One run's time-stepping state (advanced by ``SolutionLoop``)."""

    # -- the step (StepView read surface) ---------------------------------
    value: float  # current time (transient) or iteration index (steady)
    delta_t: float
    end_time: float
    index: int = 0
    write_time: bool = False

    # -- bounds / old-time bookkeeping ------------------------------------
    start_time: float = 0.0
    delta_t0: float = 0.0
    delta_t_save: float = 0.0

    # -- write-control parameters (drive the write_time decision) ---------
    write_control: str = "timeStep"
    write_interval: float = 1.0
    write_time_index: int = 0

    # -- formatting -------------------------------------------------------
    precision: int = 6
