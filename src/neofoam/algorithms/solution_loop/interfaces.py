# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The solution loop's two model-owned gather points.

``solutionLoop`` (the core loop Model) **owns** both interfaces: ``timeStepConstraint``
folds float deltaT limits with ``min`` (empty -> VGREAT = no opinion / fixed step) and
``loopCondition`` folds :class:`ConditionVote` stop verdicts via
:func:`fold_conditions` (empty -> not satisfied = keep running). Models extend them
with ``@<model>.contributes(<iface>)``; the loop consumes them by typing an
``@solutionLoop.operation`` parameter with the interface and calling it.

Leaf module: imports only ``neofoam.framework.model`` plus the stdlib-only
``conditions`` leaf, so ``time_integration`` / ``solution_loop`` can import ``VGREAT``
+ the handles without a cycle and without pulling in pybFoam.
"""

from typing import Iterable

from neofoam.algorithms.solution_loop.conditions import (
    ConditionVote,
    fold_conditions,
)
from neofoam.framework.model import Model

VGREAT = 1e300

solutionLoop = Model("solutionLoop")


@solutionLoop.interface
def timeStepConstraint(limits: Iterable[float]) -> float:
    return min(limits, default=VGREAT)


@solutionLoop.interface
def loopCondition(votes: Iterable[ConditionVote]) -> ConditionVote:
    return fold_conditions(votes)
