# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Injectable deltaT stability constraints — a plugin family.

The next ``deltaT`` is the **minimum over every constraint that models inject**
(the flow model's Courant limit, a VoF interface Courant, a ``maxDeltaT`` cap,
…), then a growth clamp — mirroring how OpenFOAM solvers stack ``maxCo`` /
``maxAlphaCo`` / ``maxDeltaT``. :class:`DeltaTConstraint` is a
:class:`~neofoam.core.plugin_system.PluginSystem` interface: any model adds a new
stability criterion by registering a ``@DeltaTConstraint.register`` class
(discriminated by ``constraint_type``); the loop just takes the ``min``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterable, Literal, Optional

from pydantic import Field

from neofoam.core.plugin_system import PluginSystem
from neofoam.io import BaseConfig

SMALL = 1e-15
VGREAT = 1e300


@PluginSystem.register(
    discriminator_variable="constraint", discriminator="constraint_type"
)
class DeltaTConstraint(BaseConfig):
    """Plugin interface: largest deltaT a model permits (``VGREAT`` = no limit).

    Concrete constraints register with :meth:`DeltaTConstraint.register`; the
    ``constraint_type`` literal discriminates them. ``measures`` names the live
    quantity the constraint reads back via ``ctx.measured(<name>)`` — published
    each step by the matching ``measurement_provider.<name>`` model. ``None``
    means the constraint needs no measurement (e.g. a constant cap).
    """

    measures: ClassVar[Optional[str]] = None

    def max_delta_t(self, ctx: Any) -> float:
        raise NotImplementedError


@DeltaTConstraint.register
class MaxDeltaTConstraint(DeltaTConstraint):
    """A constant cap (``controlDict`` ``maxDeltaT``)."""

    constraint_type: Literal["maxDeltaT"] = "maxDeltaT"
    maxDeltaT: float = Field(gt=0.0)

    def max_delta_t(self, ctx: Any) -> float:
        return self.maxDeltaT


@DeltaTConstraint.register
class CourantConstraint(DeltaTConstraint):
    """Flow stability: ``deltaT * maxCo / Co`` (the classic CFL limit)."""

    constraint_type: Literal["courant"] = "courant"
    measures: ClassVar[Optional[str]] = "courant"
    maxCo: float = Field(gt=0.0)

    def max_delta_t(self, ctx: Any) -> float:
        co = float(ctx.measured("courant"))
        if co <= SMALL:
            return VGREAT
        return float(ctx.current_delta_t()) * (self.maxCo / co)


def next_delta_t(
    constraints: Iterable[DeltaTConstraint],
    ctx: Any,
    *,
    current_dt: float,
    growth_cap: float = 1.2,
) -> float:
    """Smallest deltaT allowed by all injected constraints, clamped for growth.

    With no constraints the step is left unchanged (fixed / iteration mode).
    """
    limits = [c.max_delta_t(ctx) for c in constraints]
    if not limits:
        return current_dt
    return min(min(limits), growth_cap * current_dt)
