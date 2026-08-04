# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The extension seams of the pressure-velocity operations.

A model hooks into ``UEqn.H`` / ``pEqn.H`` with
``@<model>.contributes(<extension>.<hook>)``; the algorithms call each hook once
on the injected handle, so no operation has to know which models a case
activated. A model that owns a *single* value folded by one rule wants
``@<model>.interface`` instead. Hooks are per operation, so a contribution only
ever sees the hooks of the operation it extends; the contributions themselves
live next to each model's spec (``neofoam.mrf``, ``neofoam.fv_options``).

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

    The seed is the empty source native's ``fvOptions(U)`` starts from. A
    contribution's term joins with ``+``; a source belongs on native's
    right-hand side (``== source``), so return it as ``negated(source)``.
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
    """Transform the ddt flux correction before it joins ``phiHbyA``; the last
    active contribution wins, the unfiltered correction when there is none."""
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
    handled them, else the operation falls back to plain ``constrainPressure``."""
    return any(handled)


@pressure_extension.defines  # type: ignore[no-redef]
def correct(U: volVectorField) -> None:  # noqa: F811 — same hook name, another extension
    """Correct the velocity after the pressure corrector overwrote it."""


@mesh_update_extension.defines
def on_mesh_change() -> None:
    """React to a mesh move that changed the topology."""
