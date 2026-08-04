# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The extension seams of the VoF pressure-velocity operations.

A model hooks into interFoam's ``UEqn.H`` / ``pEqn.H`` with
``@<model>.contributes(<extension>.<hook>)``; the algorithm calls each hook once
on the injected handle, so no operation has to know which models a case
activated. Hooks are per operation, and the contributions themselves live next to
each model's spec (``neofoam.mrf``, ``neofoam.fv_options``).

These are the *mass-weighted* twins of the single-phase seams in
``solver/incompressibleFluid/models/pressure_velocity/extension.py`` — momentum
hooks carry the mixture density, pressure hooks the buoyant pressure ``p_rgh``
and the *face* mobility ``rAUf`` — so a model contributes once per solver.

Example::

    @myModel.contributes(momentum_extension.terms)
    def my_momentum_term(
        rho: volScalarField, U: volVectorField, my_runtime: Annotated[Any, "models"]
    ) -> Any:
        return my_runtime.term(rho, U)
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
def terms(rho: volScalarField, U: volVectorField, contributions: list[Any]) -> Any:
    """Terms folded into the momentum sum at ``+ ext.terms(rho, U)``.

    The seed is the empty source native's ``fvOptions(rho, U)`` starts from, so
    its rate carries the density. A contribution's term joins with ``+``; a
    source belongs on native's right-hand side (``== source``), so return it as
    ``negated(source)``.
    """
    zero_rate = pyf.dimensionedScalar("0", pyf.dimensionSet(1, -3, -1, 0, 0, 0, 0), 0.0)
    return fold(fvm.Sp(zero_rate, U), contributions)


@momentum_extension.defines
def constrain(UEqn: fvVectorMatrix) -> None:
    """Apply constraints to the relaxed momentum equation."""


@momentum_extension.defines
def correct(U: volVectorField) -> None:
    """Correct the velocity after it has been updated."""


@pressure_extension.defines
def filter_ddt_corr(corr: Any, filtered: list[Any]) -> Any:
    """Transform the rho-weighted ddt flux correction before it joins
    ``phiHbyA``; the last active contribution wins, the unfiltered correction
    when there is none."""
    return filtered[-1] if filtered else corr


@pressure_extension.defines
def make_relative(phiHbyA: surfaceScalarField) -> None:
    """Take the predicted flux relative to whatever frame this model adds."""


@pressure_extension.defines
def constrain_pressure(
    p_rgh: volScalarField,
    U: volVectorField,
    phiHbyA: surfaceScalarField,
    rAUf: surfaceScalarField,
    handled: list[bool],
) -> bool:
    """Constrain the buoyant-pressure boundaries; a contribution returns ``True``
    if it handled them, else the operation falls back to ``constrainPressure``."""
    return any(handled)


@pressure_extension.defines  # type: ignore[no-redef]
def correct(U: volVectorField) -> None:  # noqa: F811 — same hook name, another extension
    """Correct the velocity after the pressure corrector overwrote it."""


@mesh_update_extension.defines
def on_mesh_change() -> None:
    """React to a mesh move that changed the topology."""
