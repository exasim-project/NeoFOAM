# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The solution loop's model-owned gather points.

``solutionLoop`` (the core loop Model) **owns** all four interfaces. Three of them
fold float deltaT limits with ``min`` (empty -> VGREAT = no opinion), one per
OpenFOAM source they mirror, because each limit is approached differently:

* ``timeStepConstraint`` — the Courant-style limit of ``setDeltaT.H``: reached
  immediately when shrinking, damped when growing.
* ``maxTimeStep`` — the hard ``maxDeltaT`` ceiling of ``setDeltaT.H``, clipped
  *after* the damping (``min(deltaTFact*deltaT, maxDeltaT)``), never through it.
* ``initialTimeStepConstraint`` — the undamped first-step limit of
  ``setInitialDeltaT.H``, which is gated on a non-quiescent flow and can only lower
  the step.

``loopCondition`` folds bool continue-flags with ``all`` (empty -> True = keep
running). Models extend them with ``@<model>.contributes(<iface>)``; the loop
consumes them by typing an ``@solutionLoop.operation`` parameter with the interface
and calling it.

Leaf module: imports only ``neofoam.framework.model`` so ``time_integration`` /
``solution_loop`` can import ``VGREAT`` + the handles without a cycle.
"""

from typing import Iterable

from neofoam.framework.model import Model

VGREAT = 1e300

solutionLoop = Model("solutionLoop")


@solutionLoop.interface
def timeStepConstraint(limits: Iterable[float]) -> float:
    return min(limits, default=VGREAT)


@solutionLoop.interface
def maxTimeStep(ceilings: Iterable[float]) -> float:
    return min(ceilings, default=VGREAT)


@solutionLoop.interface
def initialTimeStepConstraint(limits: Iterable[float]) -> float:
    return min(limits, default=VGREAT)


@solutionLoop.interface
def loopCondition(flags: Iterable[bool]) -> bool:
    return all(flags)
