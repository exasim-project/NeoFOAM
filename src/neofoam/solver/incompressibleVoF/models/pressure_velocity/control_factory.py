# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Factory helpers that build the Python-native PIMPLE control (VoF solver).

Reads the PIMPLE subdict of ``system/fvSolution`` via pybFoam and returns a
:class:`~neofoam.algorithms.solution_loop.control.PimpleControl` so the loop
logic stays in Python (mirrors the incompressibleFluid control factory).
"""

from typing import Any

import pybFoam as pyf

from neofoam.algorithms.solution_loop.control import PimpleControl


def _read_algorithm_dict(algorithm_name: str) -> Any:
    fv_solution = pyf.dictionary.read("system/fvSolution")
    return fv_solution.subDict(algorithm_name)


def create_pimple_control(_context: dict[str, Any]) -> PimpleControl:
    """Create a :class:`PimpleControl` from the PIMPLE subdict.

    ``nCorrectors`` defaults to 2 and values below 2 are rejected here with a
    solver-level message: the PISO pressure correction needs at least two
    corrector iterations to converge the pressure-velocity coupling
    (:class:`PimpleControl` enforces ``ge=2``).
    """
    d = _read_algorithm_dict("PIMPLE")
    n_correctors = int(d.getOrDefault[int]("nCorrectors", 2))
    if n_correctors < 2:
        raise ValueError(
            "incompressibleVoF: the PISO pressure correction requires "
            f"nCorrectors >= 2, but the fvSolution PIMPLE dict sets "
            f"nCorrectors {n_correctors}."
        )
    return PimpleControl(
        nOuterCorrectors=d.getOrDefault[int]("nOuterCorrectors", 1),
        nCorrectors=n_correctors,
        nNonOrthogonalCorrectors=d.getOrDefault[int]("nNonOrthogonalCorrectors", 0),
        momentumPredictor=d.getOrDefault[bool]("momentumPredictor", True),
        turbCorr=d.getOrDefault[bool]("turbCorr", True),
    )
