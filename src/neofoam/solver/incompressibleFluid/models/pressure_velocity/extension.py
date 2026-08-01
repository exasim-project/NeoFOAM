# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The extension seams of the pressure-velocity operations, and their implementations.

Use these to hook a model into ``UEqn.H`` / ``pEqn.H`` at the points native calls
it: subclass the interface of the operation you extend, override only the sites
the model acts at, and register a factory with ``@<model>.extends(<point>)``. The
algorithms call each site once on the injected container — ``ext.constrain(UEqn)``
fans out to every active implementation, ``+ ext.terms(U)`` folds every model's
terms into the momentum sum — so no operation has to know which models a case
activated. A model that owns a *single* value folded by one rule wants
``@<model>.interface`` / ``contributes`` instead.

One interface per operation, not one per module: ``momentum``, ``continuity`` and
``mesh_update`` each declare their own, so an implementation only ever sees the
sites of the operation it extends and a model that acts in one of them carries no
inherited no-ops from the others.

The MRF and fvOptions implementations live here rather than in ``neofoam.mrf`` /
``neofoam.fv_options``: those specs are shared with ``incompressibleVoF``, whose
frame and source terms differ (``DDt(rho, U)``, ``fvOptions(rho, U)``), so the
call sites belong to this solver's algorithms.

Example::

    @myModel.extends(momentum_extension)
    def make_my_extension(my_runtime: Annotated[Any, "models"]) -> MomentumExtension:
        return _MyExtension(my_runtime)
"""

from typing import Annotated, Any

import pybFoam as pyf
from pybFoam import (
    fvVectorMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)

from neofoam.framework.model import ExtensionPoint, negated
from neofoam.fv_options import fvOptions
from neofoam.mrf import mrf

__all__ = [
    "MeshUpdateExtension",
    "MomentumExtension",
    "PressureExtension",
    "mesh_update_extension",
    "momentum_extension",
    "pressure_extension",
]


class MomentumExtension:
    """How the momentum operation can be extended — one method per site.

    Every method is a no-op default, so an implementation overrides only the sites
    its model acts at. The instances are rebuilt on every injection and hold the
    live case objects they wrap, so they must not outlive one operation call.

    Example::

        class _MyExtension(MomentumExtension):
            def correct(self, U: volVectorField) -> None:
                self._my_runtime.correct(U)
    """

    def correct_boundary_velocity(self, U: volVectorField) -> None:
        """Set the boundary velocities the momentum coefficients are built from."""

    def terms(self, U: volVectorField) -> list[Any]:
        """Terms folded into the momentum sum at ``+ ext.terms(U)``.

        A plain term joins with ``+``; a source belongs on native's right-hand
        side (``== source``), so return it as ``negated(source)`` and it joins
        with ``-`` — the same arithmetic as native's ``==``.
        """
        return []

    def constrain(self, UEqn: fvVectorMatrix) -> None:
        """Apply constraints to the relaxed momentum equation."""

    def correct(self, U: volVectorField) -> None:
        """Correct the velocity after it has been updated."""


class PressureExtension:
    """How the continuity operation can be extended — one method per site.

    Every method is a no-op default, so an implementation overrides only the sites
    its model acts at. The instances are rebuilt on every injection and hold the
    live case objects they wrap, so they must not outlive one operation call.

    Example::

        class _MyExtension(PressureExtension):
            def make_relative(self, phiHbyA: surfaceScalarField) -> None:
                self._my_runtime.makeRelative(phiHbyA)
    """

    def filter_ddt_corr(self, corr: Any) -> Any:
        """Transform the ddt flux correction before it joins ``phiHbyA``."""
        return corr

    def make_relative(self, phiHbyA: surfaceScalarField) -> None:
        """Take the predicted flux relative to whatever frame this model adds."""

    def constrain_pressure(
        self,
        p: volScalarField,
        U: volVectorField,
        phiHbyA: surfaceScalarField,
        rAU: volScalarField,
    ) -> bool:
        """Constrain the pressure boundaries; ``True`` if this call handled it."""
        return False

    def correct(self, U: volVectorField) -> None:
        """Correct the velocity after the pressure corrector overwrote it."""


class MeshUpdateExtension:
    """How the mesh_update operation can be extended — one method per site.

    The single method is a no-op default. The instances are rebuilt on every
    injection and hold the live case objects they wrap, so they must not outlive
    one operation call.

    Example::

        class _MyExtension(MeshUpdateExtension):
            def on_mesh_change(self) -> None:
                self._my_runtime.update()
    """

    def on_mesh_change(self) -> None:
        """React to a mesh move that changed the topology."""


momentum_extension = ExtensionPoint(
    "momentum_extension", MomentumExtension, folds_into=pyf.tmp_fvVectorMatrix
)
pressure_extension = ExtensionPoint("pressure_extension", PressureExtension)
mesh_update_extension = ExtensionPoint("mesh_update_extension", MeshUpdateExtension)


class _MRFMomentumExtension(MomentumExtension):
    """The rotating-frame hooks of ``UEqn.H``."""

    def __init__(self, mrf_zones: pyf.IOMRFZoneList) -> None:
        self._mrf_zones = mrf_zones

    def correct_boundary_velocity(self, U: volVectorField) -> None:
        self._mrf_zones.correctBoundaryVelocity(U)

    def terms(self, U: volVectorField) -> list[Any]:
        return [self._mrf_zones.DDt(U)]


class _MRFPressureExtension(PressureExtension):
    """The rotating-frame hooks of ``pEqn.H``."""

    def __init__(self, mrf_zones: pyf.IOMRFZoneList) -> None:
        self._mrf_zones = mrf_zones

    def filter_ddt_corr(self, corr: Any) -> Any:
        # The ddt correction belongs to the absolute frame, so it is zeroed
        # inside the MRF cells before the flux is taken relative to the rotation.
        return self._mrf_zones.zeroFilter(corr)

    def make_relative(self, phiHbyA: surfaceScalarField) -> None:
        self._mrf_zones.makeRelative(phiHbyA)

    def constrain_pressure(
        self,
        p: volScalarField,
        U: volVectorField,
        phiHbyA: surfaceScalarField,
        rAU: volScalarField,
    ) -> bool:
        pyf.constrainPressure(p, U, phiHbyA, rAU, self._mrf_zones)
        return True


class _MRFMeshUpdateExtension(MeshUpdateExtension):
    """Re-find the zone faces after a mesh move."""

    def __init__(self, mrf_zones: pyf.IOMRFZoneList) -> None:
        self._mrf_zones = mrf_zones

    def on_mesh_change(self) -> None:
        self._mrf_zones.update()


class _FvOptionsMomentumExtension(MomentumExtension):
    """The three ``fv::options`` hooks of ``UEqn.H``."""

    def __init__(self, fv_options: pyf.fvOptions) -> None:
        self._fv_options = fv_options

    def terms(self, U: volVectorField) -> list[Any]:
        # Native's ``== fvOptions(U)``: the source joins the sum subtracted.
        return [negated(self._fv_options(U))]

    def constrain(self, UEqn: fvVectorMatrix) -> None:
        self._fv_options.constrain(UEqn)

    def correct(self, U: volVectorField) -> None:
        self._fv_options.correct(U)


class _FvOptionsPressureExtension(PressureExtension):
    """The ``fv::options`` hook of ``pEqn.H``."""

    def __init__(self, fv_options: pyf.fvOptions) -> None:
        self._fv_options = fv_options

    def correct(self, U: volVectorField) -> None:
        self._fv_options.correct(U)


@mrf.extends(momentum_extension)
def make_mrf_momentum_extension(mrf_zones: Annotated[Any, "models"]) -> MomentumExtension:
    """Wrap the case's zone list for the momentum sites."""
    return _MRFMomentumExtension(mrf_zones)


@mrf.extends(pressure_extension)
def make_mrf_pressure_extension(mrf_zones: Annotated[Any, "models"]) -> PressureExtension:
    """Wrap the case's zone list for the continuity sites."""
    return _MRFPressureExtension(mrf_zones)


@mrf.extends(mesh_update_extension)
def make_mrf_mesh_update_extension(mrf_zones: Annotated[Any, "models"]) -> MeshUpdateExtension:
    """Wrap the case's zone list for the mesh_update site."""
    return _MRFMeshUpdateExtension(mrf_zones)


@fvOptions.extends(momentum_extension)
def make_fv_options_momentum_extension(fv_options: Annotated[Any, "models"]) -> MomentumExtension:
    """Wrap the case's option list for the momentum sites."""
    return _FvOptionsMomentumExtension(fv_options)


@fvOptions.extends(pressure_extension)
def make_fv_options_pressure_extension(fv_options: Annotated[Any, "models"]) -> PressureExtension:
    """Wrap the case's option list for the continuity site."""
    return _FvOptionsPressureExtension(fv_options)
