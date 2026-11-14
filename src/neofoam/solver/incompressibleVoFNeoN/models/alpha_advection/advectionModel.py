# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The alpha-advection model family of incompressibleVoFNeoN (pick one; MULES only so far)."""

from typing import Any

from .mules import mulesNeoN


class advectionModelNeoN:
    """Required model family whose single member is the NeoN MULES scheme.

    Use it as the solver's advection family; ``incompressibleVoF``'s ``advectionModel`` is the
    pybFoam alternative. isoAdvector has no NeoN port, so every case selects MULES. Stateless.

    Example::

        spec = advectionModelNeoN.detect_and_create()
    """

    @classmethod
    def all_specs(cls) -> list[Any]:
        """Every member spec of the family, case-free."""
        return [mulesNeoN]

    @classmethod
    def detect_and_create(cls) -> Any:
        """The active member for the case in the working directory."""
        return mulesNeoN
