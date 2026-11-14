# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The pressure-velocity model family of incompressibleVoFNeoN (interFoam's PIMPLE only)."""

from typing import Any

from .pimpleAlgorithm import pimpleVoFNeoN


class PressureVelocityAlgorithmVoFNeoN:
    """Required model family whose single member is interFoam's PIMPLE on NeoN.

    Use it as the solver's pressure-velocity family; ``incompressibleVoF`` has the pybFoam
    alternative. Stateless.

    Example::

        spec = PressureVelocityAlgorithmVoFNeoN.detect_and_create()
    """

    @classmethod
    def all_specs(cls) -> list[Any]:
        """Every member spec of the family, case-free."""
        return [pimpleVoFNeoN]

    @classmethod
    def detect_and_create(cls) -> Any:
        """The active member for the case in the working directory."""
        return pimpleVoFNeoN
