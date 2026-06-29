# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``constant/turbulenceProperties`` configuration.

Reads the OpenFOAM turbulence dictionary into a validated pydantic model via the
OpenFOAM IO strategy. ``simulationType`` selects RAS / LES / laminar; the
``RAS`` / ``LES`` sub-dictionaries map to nested ``BaseModel`` sub-configs —
:class:`~neofoam.io.strategies.openfoam_strategy.OpenFOAMStrategy` recurses into
sub-dictionaries automatically when a field's type is a ``BaseModel`` subclass.

This is the only turbulence module that imports ``neofoam.io`` (hence pybFoam);
tests that load it run the OpenFOAM IO path directly.
"""

from typing import Optional

from pydantic import BaseModel

from neofoam.io import BaseConfig, IOStrategy, OF

__all__ = ["RASProperties", "LESProperties", "TurbulencePropertiesConfig"]


class RASProperties(BaseModel):
    """The ``RAS`` sub-dictionary of ``turbulenceProperties``."""

    RASModel: str
    turbulence: bool = True
    printCoeffs: bool = False


class LESProperties(BaseModel):
    """The ``LES`` sub-dictionary of ``turbulenceProperties``."""

    LESModel: str
    turbulence: bool = True
    delta: str = "cubeRootVol"


@IOStrategy(OF("constant/turbulenceProperties"))
class TurbulencePropertiesConfig(BaseConfig):
    """Top-level ``constant/turbulenceProperties`` dictionary."""

    simulationType: str
    RAS: Optional[RASProperties] = None
    LES: Optional[LESProperties] = None
