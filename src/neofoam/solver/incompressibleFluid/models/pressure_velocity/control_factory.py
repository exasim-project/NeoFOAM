# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Factory helpers that build Python-native pressure-velocity controls.

These factories read the algorithm subdict of ``system/fvSolution`` via
pybFoam and return :class:`PimpleControl` / :class:`SimpleControl`
instances from :mod:`neofoam.algorithms.solution_loop.control`. They replace direct
use of ``pybFoam.pimpleControl`` so loop logic stays in Python.
"""

from typing import Any

import pybFoam as pyf

from neofoam.algorithms.solution_loop.control import PimpleControl, SimpleControl


def _read_algorithm_dict(algorithm_name: str) -> Any:
    fv_solution = pyf.dictionary.read("system/fvSolution")
    return fv_solution.subDict(algorithm_name)


def create_pimple_control(_context: dict[str, Any]) -> PimpleControl:
    """Create a :class:`PimpleControl` from the PIMPLE (or PISO) subdict.

    pisoFoam tutorials ship a ``PISO`` block instead of ``PIMPLE``. When
    ``PIMPLE`` is absent we fall back to ``PISO``, which is a single-outer-loop
    PIMPLE: ``nOuterCorrectors`` is fixed to 1 and the remaining correction
    counts / momentum predictor are read from the PISO dict.
    """
    fv_solution = pyf.dictionary.read("system/fvSolution")
    if fv_solution.isDict("PIMPLE"):
        d = fv_solution.subDict("PIMPLE")
        n_outer_correctors = d.getOrDefault[int]("nOuterCorrectors", 1)
    elif fv_solution.isDict("PISO"):
        d = fv_solution.subDict("PISO")
        n_outer_correctors = 1
    else:
        raise ValueError(
            "incompressibleFluid: system/fvSolution has neither a PIMPLE nor a "
            "PISO block to build the pressure-velocity control from."
        )
    return PimpleControl(
        nOuterCorrectors=n_outer_correctors,
        nCorrectors=d.getOrDefault[int]("nCorrectors", 2),
        nNonOrthogonalCorrectors=d.getOrDefault[int]("nNonOrthogonalCorrectors", 0),
        momentumPredictor=d.getOrDefault[bool]("momentumPredictor", True),
        turbCorr=d.getOrDefault[bool]("turbCorr", True),
    )


def create_simple_control(_context: dict[str, Any]) -> SimpleControl:
    """Create a :class:`SimpleControl` from the SIMPLE subdict."""
    d = _read_algorithm_dict("SIMPLE")
    return SimpleControl(
        nNonOrthogonalCorrectors=d.getOrDefault[int]("nNonOrthogonalCorrectors", 0),
        momentumPredictor=d.getOrDefault[bool]("momentumPredictor", True),
        consistent=d.getOrDefault[bool]("consistent", False),
        useResidualConvergence=False,
    )
