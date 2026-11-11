# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A region's coupling model, ``ctx.models["coupling"]``: a :class:`Coupling`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from neofoam.framework.operations import IterativeOp, Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata, OperationNumber

from .data import ExchangeSide
from .loop import CouplingOp
from .transport import Tag

if TYPE_CHECKING:
    from neofoam.framework.context import Context

    from .region import RegionCoupling


class Coupling:
    """The coupling of one region: its coupled sides and the operations they need.

    A solver that supports coupling adds it in its init, before its boundary
    conditions, and each ``@coupled`` boundary condition adds itself::

        ctx.models["coupling"] = Coupling(self.coupling)
        ...
        Coupling.of(ctx).add(self)
    """

    def __init__(self, coupling: Optional[RegionCoupling]) -> None:
        self.coupling = coupling
        self._sides: list[ExchangeSide] = []
        self._operations: Optional[Operations] = None

    @staticmethod
    def of(ctx: Context) -> Coupling:
        """The coupling model of the region ``ctx``; ``ValueError`` when its solver added none."""
        coupling = ctx.models.get("coupling")
        if not isinstance(coupling, Coupling):
            raise ValueError(f"the solver of region {ctx.region} does not support coupling")
        return coupling

    @property
    def sides(self) -> tuple[ExchangeSide, ...]:
        """The region's coupled sides, in the order they were added."""
        return tuple(self._sides)

    def add(self, instance: Any) -> None:
        """Add the side of the ``@coupled`` ``instance``, before the graph is built."""
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
        side = declaration.side(instance, self.coupling.region)
        if side.patch not in self.coupling.patches:
            raise ValueError(
                f"no exchange connects the coupled side of {side.label}; a coupled boundary "
                "condition needs its exchange (system/regionCoupling.yaml: exchanges)"
            )
        if self._operations is not None:
            raise ValueError(f"{side.label} is added after the region's graph was built")
        self._sides.append(side)

    def operations(self) -> Operations:
        """The region's coupling ops, one per node, outer to inner; built once."""
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

    def graph_operations(self, ctx: Context, graph: Sequence[Operation]) -> Operations:
        """The region's coupling ops and, in a transient case, its time-step op."""
        time_step = self.time_step(ctx, graph)
        return Operations([*self.operations(), *([time_step] if time_step else [])])

    def time_step(self, ctx: Context, graph: Iterable[Operation]) -> Optional[Operation]:
        """The op ``<region>.time_step``: the group steps the smallest time step any asks for.

        It runs between ``set_time_step`` and ``increment_time``; ``None`` for an
        uncoupled or steady region.
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
                # numbered: right after set_time_step, before the ops after increment_time
                operation_number=OperationNumber(0),
                first=True,
            ),
        )

    def check_graph(self, graph: Iterable[Operation]) -> None:
        """Raise ``ValueError`` when a serial node holds the time step in the resolved ``graph``."""
        coupling = self.coupling
        if coupling is None:
            return
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
