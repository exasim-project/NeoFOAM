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

from typing import Literal, Optional

from pydantic import BaseModel, model_validator

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
    """Top-level ``constant/turbulenceProperties`` dictionary.

    ``simulationType`` is the closed set ``{laminar, RAS, LES}`` so the generated
    JSON Schema advertises the three valid options as an enum — the
    :class:`~neofoam.io.strategies.openfoam_strategy.OpenFOAMStrategy` unwraps
    ``Literal[...]`` to ``str`` for disk I/O, so the on-disk format is unchanged.
    A ``model_validator`` then pins the simulationType ⇒ sub-block invariant the
    dispatcher (``selection.model_name``) silently relies on.
    """

    simulationType: Literal["laminar", "RAS", "LES"]
    RAS: Optional[RASProperties] = None
    LES: Optional[LESProperties] = None

    @model_validator(mode="after")
    def _check_simulation_type_consistency(self) -> "TurbulencePropertiesConfig":
        if self.simulationType == "laminar":
            if self.RAS is not None or self.LES is not None:
                raise ValueError(
                    "simulationType=laminar must not set RAS or LES sub-dictionary"
                )
        elif self.simulationType == "RAS":
            if self.RAS is None:
                raise ValueError("simulationType=RAS requires the RAS sub-dictionary")
            if self.LES is not None:
                raise ValueError("simulationType=RAS must not set LES sub-dictionary")
        elif self.simulationType == "LES":
            if self.LES is None:
                raise ValueError("simulationType=LES requires the LES sub-dictionary")
            if self.RAS is not None:
                raise ValueError("simulationType=LES must not set RAS sub-dictionary")
        return self
