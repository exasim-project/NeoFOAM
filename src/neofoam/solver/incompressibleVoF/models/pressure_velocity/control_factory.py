# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Factory helpers that build the Python-native PIMPLE control (VoF solver).

Reads the PIMPLE subdict of ``system/fvSolution`` via pybFoam and returns a
:class:`~neofoam.algorithms.solution_loop.control.PimpleControl` so the loop
logic stays in Python (mirrors the incompressibleFluid control factory).

When the case sets ``frozenFlow yes`` the pressure-velocity solve is switched
off entirely and the factory returns a :class:`FrozenFlowControl` instead.
"""

from typing import Any, Union

import pybFoam as pyf

from neofoam.algorithms.solution_loop.control import PimpleControl


class FrozenFlowControl:
    """Frozen-flow drop-in for :class:`PimpleControl` — no pressure-velocity solve.

    interIsoFoam runs ``if (pimple.frozenFlow()) continue;`` inside the outer
    corrector loop: alpha still advects, but the momentum predictor, the pressure
    corrector and the turbulence correction are skipped. It drives a single outer
    pass per step; ``pimpleAlgorithm``'s ``momentum``/``continuity`` recognise the
    type and return early.

    A :class:`PimpleControl` cannot stand in: the frozen tutorials pair
    ``frozenFlow yes`` with ``nCorrectors -1`` / ``nNonOrthogonalCorrectors -1``,
    which its ``ge=1``/``ge=0`` bounds reject. ``nNonOrthogonalCorrectors`` is
    carried verbatim, ``-1`` included, so the start-up flux projection runs
    ``nNonOrthCorr + 1`` — i.e. zero — passes and never solves a ``pcorr`` these
    cases declare no solver for.
    """

    def __init__(self, nNonOrthogonalCorrectors: int = 0) -> None:
        self._outer_open = True
        self.nNonOrthogonalCorrectors = nNonOrthogonalCorrectors

    def loop(self, ctx: Any = None) -> bool:
        """One outer pass per time step (re-arms on close, mirroring PimpleControl)."""
        if self._outer_open:
            self._outer_open = False
            return True
        self._outer_open = True
        return False

    def firstIter(self) -> bool:
        """The single outer pass is always the first one."""
        return True

    def finalIter(self) -> bool:
        """The single outer pass is always the final one."""
        return True

    def turbCorr(self) -> bool:
        """Never — interIsoFoam ``continue``s past the turbulence correction."""
        return False


def _read_algorithm_dict(algorithm_name: str) -> Any:
    fv_solution = pyf.dictionary.read("system/fvSolution")
    return fv_solution.subDict(algorithm_name)


def create_dynamic_mesh_controls(context: dict[str, Any]) -> dict[str, bool]:
    """The PIMPLE dict's mesh-motion switches (transcription of ``createDyMControls.H``).

    Native re-reads them every time step (``readDyMControls.H``); read once here
    because no tutorial rewrites them mid-run.
    """
    mesh = context["mesh"]
    d = _read_algorithm_dict("PIMPLE")
    return {
        "correctPhi": bool(d.getOrDefault[bool]("correctPhi", mesh.dynamic())),
        "checkMeshCourantNo": bool(d.getOrDefault[bool]("checkMeshCourantNo", False)),
        "moveMeshOuterCorrectors": bool(d.getOrDefault[bool]("moveMeshOuterCorrectors", False)),
    }


def create_pimple_control(
    _context: dict[str, Any],
) -> Union[PimpleControl, FrozenFlowControl]:
    """Create the PIMPLE control from the PIMPLE subdict.

    ``frozenFlow yes`` (default ``no``) switches the pressure-velocity solve off
    entirely and yields a :class:`FrozenFlowControl` instead.
    """
    d = _read_algorithm_dict("PIMPLE")
    if d.getOrDefault[bool]("frozenFlow", False):
        return FrozenFlowControl(
            nNonOrthogonalCorrectors=d.getOrDefault[int]("nNonOrthogonalCorrectors", 0)
        )
    return PimpleControl(
        nOuterCorrectors=d.getOrDefault[int]("nOuterCorrectors", 1),
        nCorrectors=d.getOrDefault[int]("nCorrectors", 2),
        nNonOrthogonalCorrectors=d.getOrDefault[int]("nNonOrthogonalCorrectors", 0),
        momentumPredictor=d.getOrDefault[bool]("momentumPredictor", True),
        turbCorr=d.getOrDefault[bool]("turbCorr", True),
        # pimpleControl::read() default; drives turbCorr(), which interFoam uses
        # to correct the turbulence on the final outer corrector only.
        turbOnFinalIterOnly=d.getOrDefault[bool]("turbOnFinalIterOnly", True),
    )
