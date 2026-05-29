# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``TurbulenceModel`` Protocol — the surface every turbulence model satisfies.

Both native (NeoFOAM) models and the OpenFOAM fallback adapter implement this
Protocol. It mirrors the turbulence object consumed by the incompressibleFluid
operations (``nut``/``nu`` in the Boussinesq energy equation, ``divDevReff`` in
the momentum equation, ``correct`` after pressure-velocity coupling).

Pure-Python; no pybFoam import.
"""

from typing import Any, Protocol, runtime_checkable

__all__ = ["TurbulenceModel"]


@runtime_checkable
class TurbulenceModel(Protocol):
    """Minimal turbulence model surface used by the incompressible solver."""

    def nut(self) -> Any:
        """Turbulent (eddy) viscosity field."""
        ...

    def nu(self) -> Any:
        """Laminar (molecular) viscosity field."""
        ...

    def divDevReff(self, U: Any) -> Any:
        """Divergence of the deviatoric Reynolds stress for the momentum eq."""
        ...

    def correct(self) -> None:
        """Advance the turbulence fields after pressure-velocity coupling."""
        ...
