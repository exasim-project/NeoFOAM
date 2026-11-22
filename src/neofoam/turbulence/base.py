# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``ViscousStress`` Protocol — the stress object the momentum equation calls.

The momentum predictor consumes a single collaborator: the ``viscousStress`` at
``models.viscousStress``. It exposes ``divDevReff(U)`` — the deviatoric momentum
stress term — built from the effective viscosity it refreshed (from ``nu`` and
``nut``) in its ``update`` operation before the loop. Both the native linear
assembly and the OpenFOAM-fallback delegate implement this surface.

Pure-Python; no pybFoam import.
"""

from typing import Any, Protocol, runtime_checkable

__all__ = ["ViscousStress"]


@runtime_checkable
class ViscousStress(Protocol):
    """Minimal momentum-stress surface the incompressible solver consumes."""

    def update(self, ctx: Any) -> None:
        """Refresh the effective viscosity from the Context's ``nu``/``nut``.

        Called by the momentum predictor right before :meth:`divDevReff` consumes
        it. The OpenFOAM-fallback stress owns its eddy viscosity and no-ops here.
        """
        ...

    def divDevReff(self, U: Any) -> Any:
        """Deviatoric momentum-stress divergence for the momentum equation."""
        ...
