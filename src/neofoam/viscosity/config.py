# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``constant/transportProperties`` configuration (viscosity selection).

Reads the OpenFOAM transport dictionary into a validated pydantic model via the
OpenFOAM IO strategy. ``transportModel`` selects the viscosity model
(Newtonian, CrossPowerLaw, BirdCarreau, …); ``nu`` is the kinematic viscosity
for Newtonian flow — required there, optional otherwise, since non-Newtonian
models carry their coefficients in their own sub-dictionaries instead.

This is the only viscosity module that imports ``neofoam.io`` (hence pybFoam);
tests that load it run the OpenFOAM IO path directly.
"""

from typing import Optional

from pydantic import model_validator

from neofoam.io import OF, BaseConfig, IOStrategy

__all__ = ["TransportPropertiesConfig"]


@IOStrategy(OF("constant/transportProperties"))
class TransportPropertiesConfig(BaseConfig):
    """Top-level ``constant/transportProperties`` dictionary.

    A dimensioned ``nu`` entry (``nu [ 0 2 -1 0 0 0 0 ] 1e-05``) is reduced to a
    ``float`` by the IO read path before pydantic sees it.
    """

    transportModel: str = "Newtonian"
    nu: Optional[float] = None

    @model_validator(mode="after")
    def _newtonian_needs_nu(self) -> "TransportPropertiesConfig":
        if self.transportModel == "Newtonian":
            self.newtonian_nu()
        return self

    def newtonian_nu(self) -> float:
        """Return ``nu`` for the Newtonian model, which cannot run without it."""
        if self.nu is None:
            raise ValueError(
                "transportModel Newtonian requires 'nu' in constant/transportProperties"
            )
        return self.nu
