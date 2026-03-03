# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Factory helpers for native NeoFOAM pressure-velocity controls."""

from typing import Any

import pybFoam as pyf

from neofoam.algorithms.control import PimpleControl, SimpleControl


def _read_algorithm_dict(algorithm_name: str) -> Any:
    fv_solution = pyf.dictionary.read("system/fvSolution")
    return fv_solution.subDict(algorithm_name)


def create_pimple_control(_context: dict[str, Any]) -> PimpleControl:
    """Create native `PimpleControl` from fvSolution settings."""
    d = _read_algorithm_dict("PIMPLE")

    return PimpleControl(
        nOuterCorrectors=d.getOrDefault[int]("nOuterCorrectors", 1),
        nCorrectors=d.getOrDefault[int]("nCorrectors", 2),
        nNonOrthogonalCorrectors=d.getOrDefault[int]("nNonOrthogonalCorrectors", 0),
        momentumPredictor=d.getOrDefault[bool]("momentumPredictor", True),
        turbCorr=d.getOrDefault[bool]("turbCorr", True),
    )


def create_simple_control(_context: dict[str, Any]) -> SimpleControl:
    """Create native `SimpleControl` from fvSolution settings."""
    d = _read_algorithm_dict("SIMPLE")

    return SimpleControl(
        nNonOrthogonalCorrectors=d.getOrDefault[int]("nNonOrthogonalCorrectors", 0),
        momentumPredictor=d.getOrDefault[bool]("momentumPredictor", True),
        consistent=d.getOrDefault[bool]("consistent", False),
        useResidualConvergence=False,
    )
