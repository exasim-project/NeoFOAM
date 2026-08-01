# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The extension seams of the pressure-velocity operations, and their contributions.

Use these to hook a model into ``UEqn.H`` / ``pEqn.H`` at the points native calls
it: each ``@<extension>.defines`` function below declares one site — its
parameters are what the operation passes, its body the site default — and a model
contributes to a site with ``@<model>.contributes(<extension>.<site>)``. The
algorithms call each site once on the injected handle — ``ext.constrain(UEqn)``
fans out to every active contribution, ``+ ext.terms(U)`` folds every model's
terms into the momentum sum — so no operation has to know which models a case
activated. A model that owns a *single* value folded by one rule wants
``@<model>.interface`` / ``contributes`` instead.

One extension per operation, not one per module: ``momentum``, ``continuity`` and
``mesh_update`` each declare their own, so a contribution only ever sees the
sites of the operation it extends.

The MRF and fvOptions contributions live here rather than in ``neofoam.mrf`` /
``neofoam.fv_options``: those specs are shared with ``incompressibleVoF``, whose
frame and source terms differ (``DDt(rho, U)``, ``fvOptions(rho, U)``), so the
call sites belong to this solver's algorithms.

Example::

    @myModel.contributes(momentum_extension.terms)
    def my_momentum_term(U: volVectorField, my_runtime: Annotated[Any, "models"]) -> Any:
        return my_runtime.term(U)
"""

from typing import Annotated, Any

import pybFoam as pyf
from pybFoam import (
    fvm,
    fvVectorMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)

from neofoam.framework.model import Extension, negated
from neofoam.fv_options import fvOptions
from neofoam.mrf import mrf

__all__ = [
    "mesh_update_extension",
    "momentum_extension",
    "pressure_extension",
]

momentum_extension = Extension("momentum")
pressure_extension = Extension("pressure")
mesh_update_extension = Extension("mesh_update")


# ---------------------------------------------------------------------------
# Sites — one @defines function per point the operations expose
# ---------------------------------------------------------------------------


@momentum_extension.defines
def correct_boundary_velocity(U: volVectorField) -> None:
    """Set the boundary velocities the momentum coefficients are built from."""


@momentum_extension.defines
def terms(U: volVectorField) -> Any:
    """Terms folded into the momentum sum at ``+ ext.terms(U)``.

    The seed is the empty momentum source native's ``fvOptions(U)`` starts from
    (zero matrix, ``dimVol * [U] / dimTime``). A contribution's term joins with
    ``+``; a source belongs on native's right-hand side (``== source``), so
    return it as ``negated(source)`` and it joins with ``-``.
    """
    zero_rate = pyf.dimensionedScalar("0", pyf.dimensionSet(0, 0, -1, 0, 0, 0, 0), 0.0)
    return fvm.Sp(zero_rate, U)


@momentum_extension.defines
def constrain(UEqn: fvVectorMatrix) -> None:
    """Apply constraints to the relaxed momentum equation."""


@momentum_extension.defines
def correct(U: volVectorField) -> None:
    """Correct the velocity after it has been updated."""


@pressure_extension.defines
def filter_ddt_corr(corr: Any) -> None:
    """Transform the ddt flux correction before it joins ``phiHbyA``.

    Each contribution receives the operation's unfiltered correction and returns
    its filtered form; the operation adopts the results in registration order.
    """


@pressure_extension.defines
def make_relative(phiHbyA: surfaceScalarField) -> None:
    """Take the predicted flux relative to whatever frame this model adds."""


@pressure_extension.defines
def constrain_pressure(
    p: volScalarField,
    U: volVectorField,
    phiHbyA: surfaceScalarField,
    rAU: volScalarField,
) -> None:
    """Constrain the pressure boundaries; a contribution returns ``True`` if it
    handled them (the operation falls back to native's plain
    ``constrainPressure`` when none did)."""


@pressure_extension.defines  # type: ignore[no-redef]
def correct(U: volVectorField) -> None:  # noqa: F811 — same site name, another extension
    """Correct the velocity after the pressure corrector overwrote it."""


@mesh_update_extension.defines
def on_mesh_change() -> None:
    """React to a mesh move that changed the topology."""


# ---------------------------------------------------------------------------
# MRF contributions — the rotating-frame hooks of UEqn.H / pEqn.H
# ---------------------------------------------------------------------------


@mrf.contributes(momentum_extension.correct_boundary_velocity)
def mrf_correct_boundary_velocity(U: volVectorField, mrf_zones: Annotated[Any, "models"]) -> None:
    mrf_zones.correctBoundaryVelocity(U)


@mrf.contributes(momentum_extension.terms)
def mrf_frame_acceleration(U: volVectorField, mrf_zones: Annotated[Any, "models"]) -> Any:
    return mrf_zones.DDt(U)


@mrf.contributes(pressure_extension.filter_ddt_corr)
def mrf_filter_ddt_corr(corr: Any, mrf_zones: Annotated[Any, "models"]) -> Any:
    # The ddt correction belongs to the absolute frame, so it is zeroed
    # inside the MRF cells before the flux is taken relative to the rotation.
    return mrf_zones.zeroFilter(corr)


@mrf.contributes(pressure_extension.make_relative)
def mrf_make_relative(phiHbyA: surfaceScalarField, mrf_zones: Annotated[Any, "models"]) -> None:
    mrf_zones.makeRelative(phiHbyA)


@mrf.contributes(pressure_extension.constrain_pressure)
def mrf_constrain_pressure(
    p: volScalarField,
    U: volVectorField,
    phiHbyA: surfaceScalarField,
    rAU: volScalarField,
    mrf_zones: Annotated[Any, "models"],
) -> bool:
    pyf.constrainPressure(p, U, phiHbyA, rAU, mrf_zones)
    return True


@mrf.contributes(mesh_update_extension.on_mesh_change)
def mrf_on_mesh_change(mrf_zones: Annotated[Any, "models"]) -> None:
    mrf_zones.update()


# ---------------------------------------------------------------------------
# fvOptions contributions — the fv::options hooks of UEqn.H / pEqn.H
# ---------------------------------------------------------------------------


@fvOptions.contributes(momentum_extension.terms)
def fv_options_momentum_source(U: volVectorField, fv_options: Annotated[Any, "models"]) -> Any:
    # Native's ``== fvOptions(U)``: the source joins the sum subtracted.
    return negated(fv_options(U))


@fvOptions.contributes(momentum_extension.constrain)
def fv_options_constrain(UEqn: fvVectorMatrix, fv_options: Annotated[Any, "models"]) -> None:
    fv_options.constrain(UEqn)


@fvOptions.contributes(momentum_extension.correct)
def fv_options_momentum_correct(U: volVectorField, fv_options: Annotated[Any, "models"]) -> None:
    fv_options.correct(U)


@fvOptions.contributes(pressure_extension.correct)
def fv_options_pressure_correct(U: volVectorField, fv_options: Annotated[Any, "models"]) -> None:
    fv_options.correct(U)
