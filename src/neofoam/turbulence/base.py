# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``TurbulenceModel`` Protocol — the one small read interface the solver uses.

The momentum-transport model the solver consumes exposes a single method: the
deviatoric momentum-stress term ``divDevReff(U, nu, nut)``. The molecular ``nu``
and eddy ``nut`` viscosities are Context fields the viscosity and turbulence
models own (registered/updated by their operations); the model only assembles
the stress from them. Field updates are operations, so there is no ``correct``.

Pure-Python; no pybFoam import.
"""

from typing import Any, Protocol, runtime_checkable

__all__ = ["TurbulenceModel"]


@runtime_checkable
class TurbulenceModel(Protocol):
    """Minimal momentum-transport surface used by the incompressible solver."""

    def divDevReff(self, U: Any, nu: Any, nut: Any) -> Any:
        """Deviatoric momentum-stress divergence, built from ``nu`` and ``nut``."""
        ...
