# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Time-step stability criteria for transient solvers."""

from typing import Optional

from pybFoam import Info, computeCFLNumber, dictionary

from neofoam.framework.context import Context


class CFLCondition:
    """Adjust ``deltaT`` from the current CFL number when adjustTimeStep is on.

    Reads ``adjustTimeStep`` / ``maxCo`` / ``maxDeltaT`` from
    ``system/controlDict``. Calling the instance with a :class:`Context`
    computes the CFL on ``ctx.fields["phi"]`` and updates the runtime's
    deltaT, capped by ``maxDeltaT`` and a growth ratio of 1.2.
    """

    GREAT: float = 1e30
    SMALL: float = 1e-15

    def __init__(self, maxDeltaT: Optional[float] = None) -> None:
        controlDict = dictionary.read("system/controlDict")
        try:
            self.adjustable: bool = controlDict.get[bool]("adjustTimeStep")
        except KeyError:
            self.adjustable = False

        try:
            self.maxCFL: float = controlDict.get[float]("maxCo")
        except KeyError:
            self.maxCFL = self.GREAT

        if maxDeltaT is None:
            try:
                self.maxDeltaT: float = controlDict.get[float]("maxDeltaT")
            except KeyError:
                self.maxDeltaT = 1e5
        else:
            self.maxDeltaT = maxDeltaT

        self.maxRatio: float = 1.2

    def __call__(self, ctx: Context) -> None:
        if not self.adjustable:
            return

        runTime = ctx.runtime
        phi = ctx.fields["phi"]

        deltaT = runTime.deltaTValue()
        maxCFLNumber, meanCFLNumber = computeCFLNumber(phi)

        ratio = self.maxCFL / maxCFLNumber if maxCFLNumber > self.SMALL else self.GREAT
        ratio = min(ratio, self.maxRatio)

        Info(f"deltaT = {runTime.deltaTValue()}")
        Info(f"Courant Number mean: {meanCFLNumber}, max: {maxCFLNumber}")
        finalDeltaT = min(deltaT * ratio, self.maxDeltaT)
        runTime.setDeltaT(finalDeltaT)
