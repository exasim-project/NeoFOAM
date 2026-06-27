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
    """Create a :class:`PimpleControl` from the PIMPLE subdict."""
    d = _read_algorithm_dict("PIMPLE")
    return PimpleControl(
        nOuterCorrectors=d.getOrDefault[int]("nOuterCorrectors", 1),
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


def read_residual_control(algo_dict: Any) -> dict[str, float]:
    """Read the ``residualControl`` of a PIMPLE/SIMPLE subdict.

    Handles both OpenFOAM forms — the scalar shorthand (``U 1e-3``) and the
    sub-dict form (``{ tolerance; relTol }``) — and returns ``{field: tolerance}``
    keyed by the literal ``residualControl`` entry (regex keys like ``"(k|epsilon)"``
    are kept verbatim). Only the absolute ``tolerance`` is honored; ``relTol`` is
    ignored. An absent ``residualControl`` yields an empty mapping.
    """
    if not algo_dict.found("residualControl"):
        return {}
    rc = algo_dict.subDict("residualControl")
    out: dict[str, float] = {}
    for word in rc.toc():
        name = str(word)
        if rc.isDict(name):
            sub = rc.subDict(name)
            if not sub.found("tolerance"):
                raise ValueError(
                    f"residualControl entry '{name}' is a sub-dict without a "
                    "'tolerance' entry"
                )
            out[name] = float(sub.get_scalar("tolerance"))
        else:
            out[name] = float(rc.get_scalar(name))
    return out
