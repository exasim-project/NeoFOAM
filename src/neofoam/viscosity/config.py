# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``constant/transportProperties`` configuration (viscosity selection).

Reads the OpenFOAM transport dictionary into a validated pydantic model via the
OpenFOAM IO strategy. ``transportModel`` selects the viscosity model
(Newtonian, CrossPowerLaw, BirdCarreau, …); ``nu`` is the kinematic viscosity
for Newtonian flow (optional, since non-Newtonian models carry their
coefficients in their own sub-dictionaries instead).

This is the only viscosity module that imports ``neofoam.io`` (hence pybFoam);
tests that load it run the OpenFOAM IO path directly.
"""

from typing import Optional

from neofoam.io import BaseConfig, IOStrategy, OF

__all__ = ["TransportPropertiesConfig"]


@IOStrategy(OF("constant/transportProperties"))
class TransportPropertiesConfig(BaseConfig):
    """Top-level ``constant/transportProperties`` dictionary."""

    transportModel: str = "Newtonian"
    nu: Optional[float] = None
