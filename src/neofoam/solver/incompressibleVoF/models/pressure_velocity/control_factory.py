# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Factory helpers that build the Python-native PIMPLE control (VoF solver).

Reads the PIMPLE subdict of ``system/fvSolution`` through its typed config and
returns a :class:`~neofoam.algorithms.solution_loop.control.PimpleControl` so the
loop logic stays in Python (mirrors the incompressibleFluid control factory).

When the case sets ``frozenFlow yes`` the pressure-velocity solve is switched
off entirely and the factory returns a :class:`FrozenFlowControl` instead.
"""

from typing import Any, Union

from neofoam.algorithms.solution_loop.control import PimpleControl
from neofoam.foam.algorithm_configs import DynamicMeshControls, PimpleAlgorithmConfig
from neofoam.io import OF, IOStrategy

__all__ = [
    "FrozenFlowControl",
    "VofPimpleAlgorithmConfig",
    "create_dynamic_mesh_controls",
    "create_pimple_control",
]


@IOStrategy(OF("system/fvSolution", subdict="PIMPLE"))
class VofPimpleAlgorithmConfig(PimpleAlgorithmConfig):
    """The ``PIMPLE`` block as interFoam reads it: the shared keys plus ``frozenFlow``.

    ``frozenFlow yes`` skips the pressure-velocity solve entirely, so the
    corrector counts are free to carry the ``-1`` sentinels the interIsoFoam
    tutorials pair with it (see :class:`FrozenFlowControl`). The inherited
    ``finalOnLastPimpleIterOnly`` is unused here — this port selects the
    ``<field>Final`` settings from ``pimple_control.finalIter()`` directly.
    """

    frozenFlow: bool = False


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


def create_dynamic_mesh_controls(context: dict[str, Any]) -> DynamicMeshControls:
    """The PIMPLE dict's mesh-motion switches (transcription of ``createDyMControls.H``).

    Native re-reads them every time step (``readDyMControls.H``); read once here
    because no tutorial rewrites them mid-run. ``correctPhi`` defaults to
    ``mesh.dynamic()``, resolved here where the mesh is in hand.
    """
    controls = DynamicMeshControls.load(validate=False)
    return controls.resolved(mesh_dynamic=bool(context["mesh"].dynamic()))


def create_pimple_control(
    _context: dict[str, Any],
) -> Union[PimpleControl, FrozenFlowControl]:
    """Create the PIMPLE control from the PIMPLE subdict.

    ``frozenFlow yes`` (default ``no``) switches the pressure-velocity solve off
    entirely and yields a :class:`FrozenFlowControl` instead.
    """
    config = VofPimpleAlgorithmConfig.load(validate=False)
    if config.frozenFlow:
        return FrozenFlowControl(nNonOrthogonalCorrectors=config.nNonOrthogonalCorrectors)
    return PimpleControl(
        nOuterCorrectors=config.nOuterCorrectors,
        nCorrectors=config.nCorrectors,
        nNonOrthogonalCorrectors=config.nNonOrthogonalCorrectors,
        momentumPredictor=config.momentumPredictor,
        turbCorr=config.turbCorr,
        # pimpleControl::read() default; drives turbCorr(), which interFoam uses
        # to correct the turbulence on the final outer corrector only.
        turbOnFinalIterOnly=config.turbOnFinalIterOnly,
    )
