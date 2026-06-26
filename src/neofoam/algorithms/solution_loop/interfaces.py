# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The solution loop's two gather-point interfaces.

``timeStepConstraint`` folds float deltaT limits with ``min`` (empty → VGREAT,
i.e. no opinion / fixed step); ``loopCondition`` folds bool continue-flags with
``all`` (empty → True, i.e. keep running). Contributions are registered by
models via ``@timeStepConstraint.contribute`` / ``@loopCondition.contribute``;
the consumer (``solutionLoop``) injects the spec instances and calls them.

Pure-Python: no pybFoam import (backend CFL contributions live solver-side).
"""

from typing import Iterable

from neofoam.framework.interface import Interface, InterfaceSpec

VGREAT = 1e300

timeStepConstraint: InterfaceSpec[float] = Interface("timeStepConstraint")
loopCondition: InterfaceSpec[bool] = Interface("loopCondition")


@timeStepConstraint.combine
def fold_min(limits: Iterable[float]) -> float:
    return min(limits, default=VGREAT)


@loopCondition.combine
def fold_all(flags: Iterable[bool]) -> bool:
    return all(flags)
