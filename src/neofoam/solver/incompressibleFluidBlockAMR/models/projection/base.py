# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Projection (pressure-velocity) family dispatcher for the blockAMR solver.

The block-structured engine couples pressure and velocity by a fractional-step
**projection**, not an OpenFOAM PIMPLE/PISO loop. Only one member exists —
:data:`chorinProjection` (MAC + nodal projection via
``DSLIncompressibleSolver``) — so detection is trivial; the family shape matches
the other solvers so ``solver.models(ProjectionAlgorithm, required=True)`` and
``configurations()`` work uniformly.
"""

from pathlib import Path
from typing import Any, Optional

from .chorinProjection import chorinProjection


class ProjectionAlgorithm:
    """Dispatcher for blockAMR pressure-velocity coupling algorithms."""

    @staticmethod
    def all_specs() -> list[Any]:
        """Every member spec of the family, case-free (no detection)."""
        return [chorinProjection]

    @staticmethod
    def detect_and_create(case_dir: Optional[Path] = None) -> Any:
        """Return the projection algorithm to use (only Chorin is supported)."""
        chorinProjection.algorithm_type = "Chorin"  # type: ignore[attr-defined]
        return chorinProjection

    @classmethod
    def create(cls, *, algorithm_type: str) -> Any:
        """Programmatically create the projection algorithm model."""
        if algorithm_type not in {"Chorin", "chorin", "ChorinProjection"}:
            raise ValueError(
                "incompressibleFluidBlockAMR only supports the Chorin projection; "
                f"requested {algorithm_type!r}."
            )
        chorinProjection.algorithm_type = "Chorin"  # type: ignore[attr-defined]
        return chorinProjection
