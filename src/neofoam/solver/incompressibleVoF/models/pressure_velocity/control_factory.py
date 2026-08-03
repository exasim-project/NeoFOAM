# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Factory helpers that build the Python-native PIMPLE control (VoF solver).

Reads the PIMPLE subdict of ``system/fvSolution`` via pybFoam and returns a
:class:`~neofoam.algorithms.solution_loop.control.PimpleControl` so the loop
logic stays in Python (mirrors the incompressibleFluid control factory).

When the case sets ``frozenFlow yes`` the pressure-velocity solve is switched
off entirely (interIsoFoam's ``if (pimple.frozenFlow()) continue;``); the
factory then returns a :class:`FrozenFlowControl` instead of a
:class:`PimpleControl` — see its docstring.
"""

from typing import Any, Union

import pybFoam as pyf

from neofoam.algorithms.solution_loop.control import PimpleControl


class FrozenFlowControl:
    """Frozen-flow drop-in for :class:`PimpleControl` — no pressure-velocity solve.

    interIsoFoam runs ``if (pimple.frozenFlow()) continue;`` inside the outer
    corrector loop: alpha still advects and ``mixture.correct()`` still runs, but
    the momentum predictor, the whole pressure-corrector loop and the turbulence
    correction are skipped. The frozen-flow tutorials advertise this by pairing
    ``frozenFlow yes`` with ``nCorrectors -1`` / ``nNonOrthogonalCorrectors -1``.

    We therefore build **no** :class:`PimpleControl` (whose ``ge=1`` / ``ge=0``
    bounds would reject the ``-1`` sentinels with a ``ValidationError``). This
    control only drives the PIMPLE outer loop for a single pass so alpha
    advection runs once per step; ``pimpleAlgorithm``'s ``momentum`` /
    ``continuity`` recognise it (``isinstance``) and return early, so no
    ``UEqn`` is ever assembled and the frozen tutorials' omission of the
    momentum divSchemes (e.g. ``div(rhoPhi,U)``) is honoured rather than fatal.

    ``nNonOrthogonalCorrectors`` is carried verbatim, ``-1`` included: the
    start-up flux projection (``initCorrectPhi.H``) runs for a frozen case too,
    and native's ``correctNonOrthogonal()`` executes ``nNonOrthCorr + 1`` passes
    — i.e. none at all for ``-1``. Clamping it to 0 would solve a ``pcorr``
    equation these cases do not even declare a solver for.
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

    ``correctPhi`` defaults to ``mesh.dynamic()``, the other two to ``False`` —
    exactly the native defaults, so a static case reads all three as ``False``.
    Native re-reads them every time step (``readDyMControls.H``); they are read
    once here because no tutorial rewrites them mid-run.
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

    ``nCorrectors`` is read straight from the case's PIMPLE/PISO dict and
    defaults to 2 when omitted (mirrors the incompressibleFluid control
    factory). Real interFoam / interIsoFoam cases that drive the pressure
    correction with a single corrector (``nCorrectors 1``) are honoured rather
    than rejected.

    ``frozenFlow yes`` (default ``no``) switches the pressure-velocity solve off
    entirely: a :class:`FrozenFlowControl` is returned so the tutorials'
    deliberate ``nCorrectors -1`` never reaches ``PimpleControl``'s ``ge=1``
    bound (mirrors interIsoFoam's ``if (pimple.frozenFlow()) continue;``).
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
