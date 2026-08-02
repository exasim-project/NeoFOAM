# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MRF (multiple reference frame) rotating zones, as an optional model.

Use it when a case drives its flow through a rotating cell zone declared in
``constant/MRFProperties`` (``simpleFoam/mixerVessel2D``,
``interFoam/laminar/mixerVessel2D``, …). The model is *detected*: without that
file it is never instantiated, nothing lands on the Context, and every
pressure-velocity algorithm assembles exactly the equations it assembled before
this model existed.

The model owns one runtime object, ``ctx.models["mrf_zones"]`` — OpenFOAM's own
:class:`Foam::IOMRFZoneList`, which reads the dictionary, holds the zones and
implements the frame terms. The incompressibleFluid algorithms consume it through
the ``momentum_extension`` / ``pressure_extension`` / ``mesh_update_extension``
extensions their operations define — this spec's contributions to those hooks
live at the bottom of this module; incompressibleVoF still takes it as an
*optional* injected model and branches on ``None``.

Shared by ``incompressibleFluid`` and ``incompressibleVoF``: each solver's model
package registers this one spec with its own plugin family (the momentum term
differs — ``DDt(U)`` for the single-phase solvers, ``DDt(rho, U)`` for VoF — but
that lives at the call site, not here).

Example::

    from neofoam.mrf import mrf
    mrf.register_with(incompressibleFluidModel)
"""

from pathlib import Path
from typing import Annotated, Any

import pybFoam as pyf
from pybFoam import surfaceScalarField, volScalarField, volVectorField
from pydantic import ConfigDict

from neofoam.framework.initialization import model
from neofoam.framework.model import Model
from neofoam.io import OF, BaseConfig, IOStrategy

__all__ = ["MRFPropertiesConfig", "mrf"]

# The dictionary is read by OpenFOAM (``IOMRFZoneList``), never by Python, so
# detection is a file-existence question and the path is case-relative — the
# solver runs with the case directory as its working directory.
_MRF_PROPERTIES = "constant/MRFProperties"


@IOStrategy(OF(_MRF_PROPERTIES))
class MRFPropertiesConfig(BaseConfig):
    """``constant/MRFProperties`` — one sub-dict per rotating zone.

    Declared so the file is part of the solver's config schema (collectible,
    savable, printable like any other case file). The zone entries stay
    free-form: a zone name maps to a sub-dict of ``cellZone`` /
    ``nonRotatingPatches`` / ``origin`` / ``axis`` / ``omega``, where ``omega``
    is a Function1 and can be a bare number or ``constant 6.28`` / ``table
    (...)``. Modelling those arms in pydantic would buy nothing: the values are
    consumed by ``IOMRFZoneList`` straight off disk.

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
def build(_config: MRFPropertiesConfig) -> list[Any]:
    """Publish the zone list on the Context as ``models.mrf_zones``.

    One step, depending only on the mesh — the zone list is built from cell
    zones and patch names, and every field-level hook (``correctBoundaryVelocity``,
    ``DDt``, ``makeRelative``, ``zeroFilter``) is applied by the pressure-velocity
    algorithm at the point native applies it, not here. Native constructs the
    list *after* ``createPhi.H``, so the initial flux is deliberately left as the
    absolute flux of ``0/U``. The name is ``mrf_zones`` rather than ``mrf``
    because the model runtime itself already occupies ``models.mrf``.
    """

    def create_mrf_zones(context: dict[str, Any]) -> pyf.IOMRFZoneList:
        return pyf.IOMRFZoneList(context["mesh"])

    return [model("mrf_zones", create_mrf_zones, depends_on=["mesh"])]


# ---------------------------------------------------------------------------
# Contributions — the rotating-frame hooks of UEqn.H / pEqn.H
# ---------------------------------------------------------------------------
# The import sits below the spec on purpose: the solver package imports this
# module back to register the spec, so by the time that import re-enters here
# mid-initialization, ``mrf`` above must already exist.
from neofoam.solver.incompressibleFluid.models.pressure_velocity.extension import (  # noqa: E402
    mesh_update_extension,
    momentum_extension,
    pressure_extension,
)


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
