# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MRF (multiple reference frame) rotating zones, as an optional model.

Use it when a case drives its flow through a rotating cell zone declared in
``constant/MRFProperties`` (``simpleFoam/mixerVessel2D``, …). The model is
*detected*: without that file it is never instantiated and the pressure-velocity
algorithms, which inject it optionally and branch on ``None``, assemble exactly the
equations they did before this model existed.

The spec owns one runtime object built on OpenFOAM's
:class:`Foam::IOMRFZoneList`, whose frame terms the algorithms apply where native
applies them. ``@build`` stashes that object on the model runtime, so the
contributions read it off ``self``, and its InitStep also publishes it on the
Context (``models.mrf_zones``) for the consumers that look it up by name. The
contributions below are the single-phase ones; ``incompressibleVoF`` registers
this same spec with its own plugin family and adds its mass-weighted twins in
:mod:`neofoam.solver.incompressibleVoF.models.mrf`.

Example::

    from neofoam.solver.incompressibleFluid.models.mrf import mrf
    mrf.register_with(incompressibleFluidModel)
"""

from pathlib import Path
from typing import Any

import pybFoam as pyf
from pybFoam import surfaceScalarField, volScalarField, volVectorField
from pydantic import ConfigDict

from neofoam.framework.initialization import model
from neofoam.framework.model import Model
from neofoam.io import OF, BaseConfig, IOStrategy

from .pressure_velocity.extension import (
    mesh_update_extension,
    momentum_extension,
    pressure_extension,
)

__all__ = ["MRFPropertiesConfig", "mrf"]

# Case-relative: the solver runs with the case directory as its working directory.
_MRF_PROPERTIES = "constant/MRFProperties"


@IOStrategy(OF(_MRF_PROPERTIES))
class MRFPropertiesConfig(BaseConfig):
    """``constant/MRFProperties`` — one sub-dict per rotating zone.

    Declared so the file is part of the solver's config schema. The zone entries
    stay free-form: their values are consumed by ``IOMRFZoneList`` straight off
    disk, and ``omega`` alone is a Function1 with several spellings.

    Example::

        MRFPropertiesConfig.load(case_dir=case).model_extra["MRF1"]["cellZone"]
    """

    model_config = ConfigDict(extra="allow")


mrf = Model("mrf").labeled("Rotating zones (MRF)")
mrf.config(MRFPropertiesConfig)


@mrf.detect
def detect_model() -> bool:
    """MRF is active exactly when the case carries ``constant/MRFProperties``."""
    return Path(_MRF_PROPERTIES).is_file()


@mrf.build
def build(self: Any) -> list[Any]:
    """Build the zone list, on this runtime and on the Context as ``models.mrf_zones``.

    Native constructs the list *after* ``createPhi.H``, so the initial flux is
    deliberately left as the absolute flux of ``0/U``. The Context name is
    ``mrf_zones`` because the model runtime itself already occupies
    ``models.mrf``; the contributions below take the zone list off ``self``.
    """

    def create_mrf_zones(context: dict[str, Any]) -> pyf.IOMRFZoneList:
        zones = pyf.IOMRFZoneList(context["mesh"])
        self.zones = zones
        return zones

    return [model("mrf_zones", create_mrf_zones, depends_on=["mesh"])]


# ---------------------------------------------------------------------------
# Contributions — the rotating-frame hooks of UEqn.H / pEqn.H
# ---------------------------------------------------------------------------


@mrf.contributes(momentum_extension.correct_boundary_velocity)
def mrf_correct_boundary_velocity(self: Any, U: volVectorField) -> None:
    self.zones.correctBoundaryVelocity(U)


@mrf.contributes(momentum_extension.terms)
def mrf_frame_acceleration(self: Any, U: volVectorField) -> Any:
    return self.zones.DDt(U)


@mrf.contributes(pressure_extension.filter_ddt_corr)
def mrf_filter_ddt_corr(self: Any, corr: Any) -> Any:
    # The ddt correction is absolute-frame, so zero it inside the MRF cells.
    return self.zones.zeroFilter(corr)


@mrf.contributes(pressure_extension.make_relative)
def mrf_make_relative(self: Any, phiHbyA: surfaceScalarField) -> None:
    self.zones.makeRelative(phiHbyA)


@mrf.contributes(pressure_extension.constrain_pressure)
def mrf_constrain_pressure(
    self: Any,
    p: volScalarField,
    U: volVectorField,
    phiHbyA: surfaceScalarField,
    rAU: volScalarField,
) -> bool:
    pyf.constrainPressure(p, U, phiHbyA, rAU, self.zones)
    return True


@mrf.contributes(mesh_update_extension.on_mesh_change)
def mrf_on_mesh_change(self: Any) -> None:
    self.zones.update()
