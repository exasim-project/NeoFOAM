# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The closed-domain pressure reference every pressure-velocity algorithm carries.

``Foam::setRefCell`` answers three things about a case: which cell pins the free
pressure level, what value it is pinned to, and whether the boundary conditions
leave the level undetermined at all. Each pressure-velocity algorithm computes
that once at initialisation and publishes it on the Context for its corrector.

It is **runtime-computed data, not a config**: nothing here is read from (or
written to) a case dictionary — ``pRefCell`` / ``pRefValue`` come from the
algorithm's ``fvSolution`` block *through* ``setRefCell``, which resolves a
``pRefPoint`` to a cell index against the live mesh. Hence a plain frozen
dataclass rather than a :class:`~neofoam.io.BaseConfig`, and hence it stays
injected by name out of ``ctx.models`` — the marker for runtime state — with
only the string keys of the old ``dict`` payload replaced by typed attributes.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PressureReference"]


@dataclass(frozen=True)
class PressureReference:
    """Where and to what the pressure level is pinned on a closed domain.

    Attributes:
        cell: The reference cell index (``pRefCell``). Negative on a rank that
            does not own the cell, and on every rank of an open domain.
        value: The level the reference cell is pinned to (``pRefValue``).
        needs_ref: Whether the boundary conditions leave the pressure level
            undetermined, i.e. whether the reference has to be applied at all.
    """

    cell: int
    value: float
    needs_ref: bool
