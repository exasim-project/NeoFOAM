# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The extension seams of the pressure-velocity operations.

Use these to hook a model into ``UEqn.H`` / ``pEqn.H`` at the points native calls
it: each ``@<extension>.defines`` function below declares one hook — its leading
parameters are what the operation passes, a trailing parameter receives the
active contributions' results and the body combines them (a declaration without
one broadcasts the raw results) — and a model contributes to a hook with
``@<model>.contributes(<extension>.<hook>)``. The algorithms call each hook once
on the injected handle — ``ext.constrain(UEqn)`` fans out to every active
contribution, ``+ ext.terms(U)`` folds every model's terms into the momentum sum
— so no operation has to know which models a case activated. A model that owns a
*single* value folded by one rule wants ``@<model>.interface`` / ``contributes``
instead.

One extension per operation, not one per module: ``momentum``, ``continuity`` and
``mesh_update`` each declare their own, so a contribution only ever sees the
hooks of the operation it extends.

This module only *defines* the hooks; each model's contributions live next to
its spec (``neofoam.mrf``, ``neofoam.fv_options``).

Example::

    @myModel.contributes(momentum_extension.terms)
    def my_momentum_term(U: volVectorField, my_runtime: Annotated[Any, "models"]) -> Any:
        return my_runtime.term(U)
"""

from typing import Any

import pybFoam as pyf
from pybFoam import (
    fvm,
    fvVectorMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)

from neofoam.framework.model import Extension, fold

__all__ = [
    "mesh_update_extension",
    "momentum_extension",
    "pressure_extension",
]

momentum_extension = Extension("momentum")
pressure_extension = Extension("pressure")
mesh_update_extension = Extension("mesh_update")


# ---------------------------------------------------------------------------
# Hooks — one @defines function per point the operations expose
# ---------------------------------------------------------------------------


@momentum_extension.defines
def correct_boundary_velocity(U: volVectorField) -> None:
    """Set the boundary velocities the momentum coefficients are built from."""


@momentum_extension.defines
def terms(U: volVectorField, contributions: list[Any]) -> Any:
    """Terms folded into the momentum sum at ``+ ext.terms(U)``.

    The fold seed is the empty momentum source native's ``fvOptions(U)`` starts
    from (zero matrix, ``dimVol * [U] / dimTime``). A contribution's term joins
    with ``+``; a source belongs on native's right-hand side (``== source``), so
    return it as ``negated(source)`` and it joins with ``-``.
    """
    zero_rate = pyf.dimensionedScalar("0", pyf.dimensionSet(0, 0, -1, 0, 0, 0, 0), 0.0)
    return fold(fvm.Sp(zero_rate, U), contributions)


@momentum_extension.defines
def constrain(UEqn: fvVectorMatrix) -> None:
    """Apply constraints to the relaxed momentum equation."""


@momentum_extension.defines
def correct(U: volVectorField) -> None:
    """Correct the velocity after it has been updated."""


@pressure_extension.defines
def filter_ddt_corr(corr: Any, filtered: list[Any]) -> Any:
    """Transform the ddt flux correction before it joins ``phiHbyA``.

    Each contribution receives the operation's unfiltered correction and returns
    its filtered form; the last active contribution's result is adopted, the
    unfiltered correction when none is.
    """
    return filtered[-1] if filtered else corr


@pressure_extension.defines
def make_relative(phiHbyA: surfaceScalarField) -> None:
    """Take the predicted flux relative to whatever frame this model adds."""


@pressure_extension.defines
def constrain_pressure(
    p: volScalarField,
    U: volVectorField,
    phiHbyA: surfaceScalarField,
    rAU: volScalarField,
    handled: list[bool],
) -> bool:
    """Constrain the pressure boundaries; a contribution returns ``True`` if it
    handled them (the operation falls back to native's plain
    ``constrainPressure`` when none did)."""
    return any(handled)


@pressure_extension.defines  # type: ignore[no-redef]
def correct(U: volVectorField) -> None:  # noqa: F811 — same hook name, another extension
    """Correct the velocity after the pressure corrector overwrote it."""


@mesh_update_extension.defines
def on_mesh_change() -> None:
    """React to a mesh move that changed the topology."""
