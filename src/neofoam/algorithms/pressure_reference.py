# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The closed-domain pressure reference every pressure-velocity algorithm carries.

Runtime-computed data, not a config: ``Foam::setRefCell`` resolves the case's
``fvSolution`` entries against the live mesh once at initialisation. Hence a
plain frozen dataclass rather than a :class:`~neofoam.io.BaseConfig`, and hence
it stays injected by name out of ``ctx.models``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PressureReference"]


@dataclass(frozen=True)
class PressureReference:
    """Where and to what the pressure level is pinned on a closed domain."""

    #: ``pRefCell``; negative on a rank that does not own it and on an open domain.
    cell: int
    value: float
    #: Whether the boundary conditions leave the pressure level undetermined.
    needs_ref: bool
