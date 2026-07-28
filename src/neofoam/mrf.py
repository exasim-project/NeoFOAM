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
implements the frame terms. The algorithms that consume it (SIMPLE and both
PIMPLEs) take it as an *optional* injected model and branch on ``None``, so this
module never has to supply a no-op stand-in.

Shared by ``incompressibleFluid`` and ``incompressibleVoF``: each solver's model
package registers this one spec with its own plugin family (the momentum term
differs — ``DDt(U)`` for the single-phase solvers, ``DDt(rho, U)`` for VoF — but
that lives at the call site, not here).

Example::

    from neofoam.mrf import mrf
    mrf.register_with(incompressibleFluidModel)
"""

from pathlib import Path
from typing import Any

import pybFoam as pyf
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
