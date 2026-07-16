# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``realizableKE`` — a **fallback-only** momentum-transport model.

NeoFOAM has no native NeoN closure for the realizable k-epsilon model (yet), so
this model registers a **name** — making it visible to selection and the MCP —
but declares **only** the pybFoam-OpenFOAM fallback backend: no ``@build`` and no
native ``@operation``, just one co-located ``fallback=True`` ``correct`` op. The
wrapped pybFoam ``incompressibleTurbulenceModel`` owns ``nut`` and its momentum
stress; the only thing NeoFOAM schedules is a single ``correct()`` after the
pressure-velocity loop.

It is usable only when a solver selects ``fallback=True`` (``incompressibleFluid``).
Asking for it on the native NeoN path (``fallback=False``) raises cleanly at
selection time — this is the reference shape for the "pybFoam-variants only"
model documented in the family-merge plan.
"""

from typing import Annotated, Any

from neofoam.framework.context import FieldUpdates

from ..config import TurbulencePropertiesConfig
from ..momentumTransport import Model, momentumTransportModel

__all__ = ["realizableKE"]

realizableKE = Model("realizableKE").register_with(momentumTransportModel)
realizableKE.config(TurbulencePropertiesConfig)
# no @build, no native @operation — fallback-only


@realizableKE.operation(name="realizableKECorrect", fallback=True)
def correct(
    self: Any,
    turbulence: Annotated[Any, "models"],  # the wrapped pybFoam handle
) -> FieldUpdates:
    """Advance k/epsilon/nut via OpenFOAM's own ``realizableKE::correct()``.

    Resolved from the Context, never captured in the closure
    (see [[project_pybfoam_op_closure_cycle]]).
    """
    turbulence.correct()
    return FieldUpdates({})
