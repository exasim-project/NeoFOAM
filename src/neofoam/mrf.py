# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MRF (multiple reference frame) rotating zones, as an optional model.

Use it when a case drives its flow through a rotating cell zone declared in
``constant/MRFProperties`` (``simpleFoam/mixerVessel2D``, …). The model is
*detected*: without that file it is never instantiated and the pressure-velocity
algorithms, which inject it optionally and branch on ``None``, assemble exactly the
equations they did before this model existed.

It owns one runtime object, ``ctx.models["mrf_zones"]`` — OpenFOAM's own
:class:`Foam::IOMRFZoneList` — whose frame terms the algorithms apply where native
applies them. The incompressibleFluid algorithms reach them through the extension
points their operations declare — this spec registers the implementations there;
incompressibleVoF still injects the model optionally and branches on ``None``.
Shared by ``incompressibleFluid`` and ``incompressibleVoF``, which
each register this one spec with their own plugin family.

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
def build(_config: MRFPropertiesConfig) -> list[Any]:
    """Publish the zone list on the Context as ``models.mrf_zones``.

    Native constructs the list *after* ``createPhi.H``, so the initial flux is
    deliberately left as the absolute flux of ``0/U``. Named ``mrf_zones`` because
    the model runtime itself already occupies ``models.mrf``.
    """

    def create_mrf_zones(context: dict[str, Any]) -> pyf.IOMRFZoneList:
        return pyf.IOMRFZoneList(context["mesh"])

    return [model("mrf_zones", create_mrf_zones, depends_on=["mesh"])]
