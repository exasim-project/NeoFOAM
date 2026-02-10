# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from typing import Optional
from pybFoam import Info, dictionary, computeCFLNumber
from foamadapter.framework.context import Context


class CFLCondition:
    """Condition for CFL-based time stepping."""

    def __init__(self, maxDeltaT: Optional[float] = None) -> None:
        self.GREAT = 1e30
        self.SMALL = 1e-15
        controlDict = dictionary.read("system/controlDict")
        self.adjustable = controlDict.get[bool]("adjustTimeStep")
        self.maxCFL = controlDict.get[float]("maxCo")

        # Read maxDeltaT from controlDict if not provided
        if maxDeltaT is None:
            try:
                self.maxDeltaT = controlDict.get[float]("maxDeltaT")
            except KeyError:
                self.maxDeltaT = 1e5  # Default value
        else:
            self.maxDeltaT = maxDeltaT

        self.maxRatio = 1.2

    def __call__(self, ctx: Context) -> None:
        """Adjust time step based on CFL number and continue."""
        runTime = ctx.runTime
        phi = ctx.fields["phi"]

        if not self.adjustable:
            return

        deltaT = runTime.deltaTValue()
        maxCFLNumber, meanCFLNumber = computeCFLNumber(phi)

        ratio = self.maxCFL / maxCFLNumber if maxCFLNumber > self.SMALL else self.GREAT
        ratio = min(ratio, self.maxRatio)

        Info(f"deltaT = {runTime.deltaTValue()}")
        Info(f"Courant Number mean: {meanCFLNumber}, max: {maxCFLNumber}")
        finalDeltaT = min(deltaT * ratio, self.maxDeltaT)
        runTime.setDeltaT(finalDeltaT)
