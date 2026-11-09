# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The coupling of one region as a model: ``ctx.models["coupling"]``, a :class:`Coupling`.

A solver that supports coupling adds it in its init, before it builds its
boundary conditions, with its runtime's
:class:`~neofoam.framework.coupling.RegionCoupling` (``None`` when the case does
not couple the region)::

    ctx.models["coupling"] = Coupling(self.coupling)

A ``@coupled`` boundary condition, or any other ``@coupled`` model, adds itself::

    @coupled
    class CoupledConduction(SlabPatch):
        def __init__(self, field, patch, ctx, value):
            super().__init__(field, patch, ctx)
            Coupling.of(ctx).add(self)

The region's execution graph then gets the model's operations: the region's
:class:`~neofoam.framework.coupling.CouplingOp`, built once from all sides; none
in a region the case does not couple.

:meth:`Coupling.agree` makes the regions of a group agree on a value, e.g. a
transient solver on its next time step::

    dt = Coupling.of(ctx).agree(dt)          # the smallest of the group
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from neofoam.framework.operations import Operation, Operations
from neofoam.framework.types import OperationMetadata, Placement

from .loop import CouplingOp, ExchangeSide

if TYPE_CHECKING:
    from neofoam.framework.context import Context

    from .region import RegionCoupling


class Coupling:
    """The coupling of one region: its :class:`RegionCoupling` and its coupled sides.

    ``coupling`` is ``None`` when the case does not couple the region.
    """

    def __init__(self, coupling: Optional[RegionCoupling]) -> None:
        #: What the region knows of the case's coupling; None: not coupled.
        self.coupling = coupling
        self._sides: list[ExchangeSide] = []
        self._operations: Optional[Operations] = None
        # the agreements so far: each gets its own tag, the same in every region
        self._agreements = 0

    @staticmethod
    def of(ctx: Context) -> Coupling:
        """The coupling model of the region ``ctx``.

        Raises ``ValueError`` when its solver added none (``ctx.models["coupling"]``).
        """
        coupling = ctx.models.get("coupling")
        if not isinstance(coupling, Coupling):
            raise ValueError(f"the solver of region {ctx.region} does not support coupling")
        return coupling

    @property
    def sides(self) -> tuple[ExchangeSide, ...]:
        """The region's coupled sides, in the order they were added."""
        return tuple(self._sides)

    def add(self, instance: Any) -> None:
        """Add the side of the ``@coupled`` ``instance`` (a boundary condition or other model).

        The instance needs ``field`` (the field it couples) and ``patch``. Raises
        ``TypeError`` when it is not ``@coupled`` or lacks one of them,
        ``ValueError`` when no exchange connects its patch (also when the case
        does not couple the region) or the graph is built.
        """
        declaration = getattr(type(instance), "__coupled__", None)
        if declaration is None:
            raise TypeError(f"{type(instance).__qualname__} is not @coupled")
        for attribute in ("field", "patch"):
            if not hasattr(instance, attribute):
                raise TypeError(
                    f"@coupled {type(instance).__qualname__} sets no self.{attribute} in __init__"
                )
        if self.coupling is None:
            raise ValueError(
                f"no exchange connects the coupled side on patch {instance.patch}: the case "
                "does not couple its region (system/regionCoupling.yaml: exchanges)"
            )
        side = ExchangeSide(
            self.coupling.region, instance.patch, instance.field, declaration.side(instance)
        )
        if side.patch not in self.coupling.patches:
            raise ValueError(
                f"no exchange connects the coupled side of {side.label}; a coupled boundary "
                "condition needs its exchange (system/regionCoupling.yaml: exchanges)"
            )
        if self._operations is not None:
            raise ValueError(f"{side.label} is added after the region's graph was built")
        self._sides.append(side)

    def operations(self) -> Operations:
        """The region's coupling op; built once, from all sides.

        None when the case does not couple the region. Raises ``ValueError`` when
        an exchange of the region has no side (also when the region has none at
        all), or a patch two.
        """
        if self._operations is None:
            if self.coupling is None:
                self._operations = Operations()
            else:
                couple = CouplingOp(self.coupling, self._sides)
                metadata = OperationMetadata(op_name=couple.label, wraps=Placement())
                self._operations = Operations([Operation(func=couple, metadata=metadata)])
        return self._operations

    def agree(self, value: Any, how: Callable[[list[Any]], Any] = min) -> Any:
        """The value the regions of the group agree on: ``how`` of all their ``value`` s.

        Every region of the group calls it, in the same order: each sends its
        value to the leader, which reduces them with ``how`` and sends the result
        to all. Blocks until the leader's result is there. A region the case does
        not couple gets its ``value`` back.
        """
        coupling = self.coupling
        if coupling is None:
            return value
        self._agreements += 1
        tag = f"agree:{self._agreements}"
        region, leader, transport = coupling.region, coupling.leader, coupling.transport
        if region != leader:
            transport.send(region, leader, tag, value)
            return transport.receive(leader, region, tag)
        values = [
            value if r == region else transport.receive(r, region, tag) for r in coupling.group
        ]
        agreed = how(values)
        for other in coupling.group:
            if other != region:
                transport.send(region, other, tag, agreed)
        return agreed
