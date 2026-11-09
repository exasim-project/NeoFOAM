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

The region's execution graph then gets the model's operations: one
:class:`~neofoam.framework.coupling.CouplingOp` per coupling node of the region
(``<region>.<node>``, outer to inner), built once from the sides of the node's
exchanges; none in a region the case does not couple.

In a transient case (the region's ``ctx.time`` is set) the graph also gets the
region's time-step op ``<region>.time_step`` (:meth:`Coupling.time_step`): the
regions of the group step the smallest time step any of them asks for. It runs
after the solutionLoop's ``set_time_step`` and before its ``increment_time``,
once per time step (``first``), and reads and writes the time step in the
region's context (``ctx.time.delta_t``). It must not end up inside a serial
coupling op: :meth:`Coupling.check` raises for the resolved graph (the
runtime's ``resolve_operations`` calls it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional

from neofoam.framework.operations import IterativeOp, Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata, OperationNumber

from .loop import CouplingOp, ExchangeSide
from .transport import Tag

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
        """The region's coupling ops, one per coupling node, outer to inner; built once.

        Each from the sides on the patches of its node's exchanges, placed as the
        node's placement for the region says. None when the case does not couple
        the region. Raises ``ValueError`` when an exchange of the region has no
        side (also when the region has none at all), a patch two, or the sides of
        one field are in two nodes.
        """
        if self._operations is None:
            coupling = self.coupling
            if coupling is None:
                self._operations = Operations()
            else:
                by_node: dict[str, list[ExchangeSide]] = {node.name: [] for node in coupling.nodes}
                in_node: dict[str, str] = {}
                for side in self._sides:
                    name = coupling.node_of(side.patch).name
                    if in_node.setdefault(side.field, name) != name:
                        raise ValueError(
                            f"field {side.field} of region {coupling.region} is coupled in the "
                            f"nodes '{in_node[side.field]}' and '{name}' ({side.label}); a "
                            "region's field is in at most one coupling node"
                        )
                    by_node[name].append(side)
                ops: list[Operation] = []
                for node in coupling.nodes:
                    couple = CouplingOp(coupling, node, by_node[node.name])
                    metadata = OperationMetadata(
                        op_name=couple.label, wraps=node.placement_of(coupling.region)
                    )
                    ops.append(Operation(func=couple, metadata=metadata))
                self._operations = Operations(ops)
        return self._operations

    def time_step(self, ctx: Context, graph: Iterable[Operation]) -> Optional[Operation]:
        """The region's time-step op ``<region>.time_step``, or ``None``.

        ``None`` when the case does not couple the region or the region does not
        step in time (``ctx.time`` is ``None``, e.g. a steady region). The op
        sends the time step the region asks for (``ctx.time.delta_t``) to the
        group's leader, which takes the smallest of the group and sends it back
        (tag ``time/dt``); each region writes it into ``ctx.time.delta_t``, where
        ``increment_time`` takes it from. It runs after ``set_time_step`` and
        before ``increment_time``, once per time step (``first``).

        Raises ``ValueError`` when ``graph`` (the region's operations, loops with
        what they hold) has no ``set_time_step`` or ``increment_time``.
        """
        coupling = self.coupling
        if coupling is None or len(coupling.group) < 2 or ctx.time is None:
            return None
        names = _names(graph)
        missing = [name for name in (SET_TIME_STEP, INCREMENT_TIME) if name not in names]
        if missing:
            raise ValueError(
                f"region {coupling.region} steps in time coupled with {list(coupling.group)}, "
                f"but its graph has no {missing}; the coupled regions agree on the time step "
                f"between the solutionLoop's {SET_TIME_STEP} and {INCREMENT_TIME}"
            )
        return Operation(
            func=SequentialOp(_TimeStep(coupling)),
            metadata=OperationMetadata(
                op_name=f"{coupling.region}.time_step",
                depends_on=[SET_TIME_STEP],
                before=[INCREMENT_TIME],
                # numbered: the resolver runs it right after set_time_step, before
                # the operations stepped after increment_time
                operation_number=OperationNumber(0),
                first=True,
            ),
        )

    def check(self, graph: Iterable[Operation]) -> None:
        """Raise ``ValueError`` when the resolved ``graph`` holds the time step in a serial node.

        A serial pass of a region waits for its partner's whole pass; when that
        holds the time-step op, which waits for the region, neither comes back.
        """
        coupling = self.coupling
        if coupling is None:
            return
        # the labels of the region's serial coupling ops, and their nodes' names
        serial = {
            f"{coupling.region}.{node.name}": node.name
            for node in coupling.nodes
            if not node.parallel
        }
        holder = _holder(graph, f"{coupling.region}.time_step", serial)
        if holder is not None:
            raise ValueError(
                f"coupling '{holder}' is serial but holds the time step of its regions; make it "
                f"parallel or place it around named operations "
                f"(wraps.{coupling.region}.operations)"
            )


# The solutionLoop operations the time-step op runs between.
SET_TIME_STEP = "set_time_step"
INCREMENT_TIME = "increment_time"


def _names(ops: Iterable[Operation]) -> set[str]:
    """The names of ``ops`` and of every operation inside their loops."""
    names: set[str] = set()
    for op in ops:
        if op.operation_name:
            names.add(op.operation_name)
        if isinstance(op.func, IterativeOp):
            names |= _names(op.sub_operations)
    return names


def _holder(ops: Iterable[Operation], name: str, serial: dict[str, str]) -> Optional[str]:
    """The node of the outermost loop of ``serial`` (by label) holding operation ``name``."""
    for op in ops:
        if isinstance(op.func, IterativeOp):
            if op.operation_name in serial and name in _names(op.sub_operations):
                return serial[op.operation_name]
            inside = _holder(op.sub_operations, name, serial)
            if inside is not None:
                return inside
    return None


class _TimeStep:
    """The time-step op of a region: its group steps the smallest time step any region asks for."""

    def __init__(self, coupling: RegionCoupling) -> None:
        self._coupling = coupling

    def __call__(self, ctx: Context) -> None:
        coupling = self._coupling
        region, leader, transport = coupling.region, coupling.leader, coupling.transport
        asked = ctx.time.delta_t
        if region != leader:
            transport.send(region, leader, Tag.TIME_DT, asked)
            agreed = transport.receive(leader, region, Tag.TIME_DT)
        else:
            agreed = min(
                asked if other == region else transport.receive(other, region, Tag.TIME_DT)
                for other in coupling.group
            )
            for other in coupling.group:
                if other != region:
                    transport.send(region, other, Tag.TIME_DT, agreed)
        ctx.time.delta_t = agreed
