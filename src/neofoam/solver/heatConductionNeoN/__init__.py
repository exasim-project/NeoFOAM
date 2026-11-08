# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""heatConductionNeoN solver package.

``laplacianFoam`` on NeoN: heat conduction ``∂T/∂t = ∇·(DT ∇T)``, with the
coupled patches ``coupledTemperature`` / ``coupledHeatFlux`` of a multi-region
case (``multiRun``).
"""

from .heatConductionNeoN import heatConductionNeoN, run

__all__ = ["heatConductionNeoN", "run"]
