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


def _pimple_dict() -> tuple[Any, bool]:
    """The PIMPLE subdict and whether it came from a ``PISO`` block instead.

    pisoFoam tutorials ship a ``PISO`` block where pimpleFoam ships ``PIMPLE``;
    PISO is a single-outer-loop PIMPLE, so it is read from the same place.
    """
    fv_solution = pyf.dictionary.read("system/fvSolution")
    if fv_solution.isDict("PIMPLE"):
        return fv_solution.subDict("PIMPLE"), False
    if fv_solution.isDict("PISO"):
        return fv_solution.subDict("PISO"), True
    raise ValueError(
        "incompressibleFluid: system/fvSolution has neither a PIMPLE nor a "
        "PISO block to build the pressure-velocity control from."
    )


def create_dynamic_mesh_controls(context: dict[str, Any]) -> dict[str, bool]:
    """The PIMPLE dict's mesh-motion switches (transcription of ``createDyMControls.H``).

    ``correctPhi`` defaults to ``mesh.dynamic()``, the other two to ``False`` —
    exactly the native defaults, so a static case reads all three as ``False``.
    Native re-reads them every time step (``readDyMControls.H``); they are read
    once here because no tutorial rewrites them mid-run.
    """
    mesh = context["mesh"]
    d, _ = _pimple_dict()
    return {
        "correctPhi": bool(d.getOrDefault[bool]("correctPhi", mesh.dynamic())),
        "checkMeshCourantNo": bool(d.getOrDefault[bool]("checkMeshCourantNo", False)),
        "moveMeshOuterCorrectors": bool(d.getOrDefault[bool]("moveMeshOuterCorrectors", False)),
    }


def create_pimple_control(_context: dict[str, Any]) -> PimpleControl:
    """Create a :class:`PimpleControl` from the PIMPLE (or PISO) subdict.

    pisoFoam tutorials ship a ``PISO`` block instead of ``PIMPLE``. When
    ``PIMPLE`` is absent we fall back to ``PISO``, which is a single-outer-loop
    PIMPLE: ``nOuterCorrectors`` is fixed to 1 and the remaining correction
    counts / momentum predictor are read from the PISO dict.
    """
    d, from_piso = _pimple_dict()
    n_outer_correctors = 1 if from_piso else d.getOrDefault[int]("nOuterCorrectors", 1)
    return PimpleControl(
        nOuterCorrectors=n_outer_correctors,
        nCorrectors=d.getOrDefault[int]("nCorrectors", 2),
        nNonOrthogonalCorrectors=d.getOrDefault[int]("nNonOrthogonalCorrectors", 0),
        momentumPredictor=d.getOrDefault[bool]("momentumPredictor", True),
        turbCorr=d.getOrDefault[bool]("turbCorr", True),
        # pimpleControl::read() defaults: turbOnFinalIterOnly true,
        # finalOnLastPimpleIterOnly false.
        turbOnFinalIterOnly=d.getOrDefault[bool]("turbOnFinalIterOnly", True),
        finalOnLastPimpleIterOnly=d.getOrDefault[bool]("finalOnLastPimpleIterOnly", False),
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
