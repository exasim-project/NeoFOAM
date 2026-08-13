# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MRF rotating zones for incompressibleVoF: the mass-weighted contributions.

Use it the same way as the single-phase model — the case declares its rotating
zones in ``constant/MRFProperties`` and nothing else changes. The *spec* is the
one defined in :mod:`neofoam.solver.incompressibleFluid.models.mrf`, imported
here and registered with both plugin families; this module only adds the
contributions on interFoam's own extension hooks, because the terms differ:
mass-weighted frame acceleration ``DDt(rho, U)``, and ``p_rgh`` constrained on
the *face* mobility ``rAUf``.

Example::

    from neofoam.solver.incompressibleVoF.models.mrf import mrf
    mrf.register_with(incompressibleVoFModel)
"""

from typing import Any

import pybFoam as pyf
from pybFoam import surfaceScalarField, volScalarField, volVectorField

from neofoam.solver.incompressibleFluid.models.mrf import mrf

from .pressure_velocity.extension import (
    mesh_update_extension,
    momentum_extension,
    pressure_extension,
)

__all__ = ["mrf"]


@mrf.contributes(momentum_extension.correct_boundary_velocity)
def mrf_vof_correct_boundary_velocity(self: Any, U: volVectorField) -> None:
    self.zones.correctBoundaryVelocity(U)


@mrf.contributes(momentum_extension.terms)
def mrf_vof_frame_acceleration(self: Any, rho: volScalarField, U: volVectorField) -> Any:
    return self.zones.DDt(rho, U)


@mrf.contributes(pressure_extension.filter_ddt_corr)
def mrf_vof_filter_ddt_corr(self: Any, corr: Any) -> Any:
    # The ddt correction is absolute-frame, so zero it inside the MRF cells.
    return self.zones.zeroFilter(corr)


@mrf.contributes(pressure_extension.make_relative)
def mrf_vof_make_relative(self: Any, phiHbyA: surfaceScalarField) -> None:
    self.zones.makeRelative(phiHbyA)


@mrf.contributes(pressure_extension.constrain_pressure)
def mrf_vof_constrain_pressure(
    self: Any,
    p_rgh: volScalarField,
    U: volVectorField,
    phiHbyA: surfaceScalarField,
    rAUf: surfaceScalarField,
) -> bool:
    pyf.constrainPressure(p_rgh, U, phiHbyA, rAUf, self.zones)
    return True


@mrf.contributes(mesh_update_extension.on_mesh_change)
def mrf_vof_on_mesh_change(self: Any) -> None:
    self.zones.update()
