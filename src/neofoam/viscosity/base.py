# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``ViscosityModel`` Protocol — the surface every viscosity model satisfies.

Both native (NeoFOAM) transport models and the OpenFOAM fallback adapter
implement this Protocol. It mirrors the transport object consumed by the
incompressibleFluid solver (the ``laminarTransport`` model passed to the
turbulence factory): ``nu`` returns the kinematic viscosity field and
``correct`` updates it for rate-dependent (non-Newtonian) models.

Pure-Python; no pybFoam import.
"""

from typing import Any, Protocol, runtime_checkable

__all__ = ["ViscosityModel"]


@runtime_checkable
class ViscosityModel(Protocol):
    """Minimal viscosity/transport model surface used by the solver."""

    def nu(self) -> Any:
        """Kinematic viscosity field."""
        ...

    def correct(self) -> None:
        """Recompute viscosity (no-op for Newtonian; strain-dependent else)."""
        ...
