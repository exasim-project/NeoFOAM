# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A region's coupling model ``ctx.models["coupling"]``, a :class:`Coupling`; :func:`coupling`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata, OperationNumber

from .data import CoupledPort, base_name
from .iteration._loop_op import CouplingOp
from .location import Location, MeshGeometry

if TYPE_CHECKING:
    from neofoam.framework.context import Context

    from .region import RegionCoupling
from .iteration._time_step import (
    INCREMENT_TIME,
    SET_TIME_STEP,
    TimeStep,
    operation_names,
    serial_holder,
)


class Coupling:
    """The coupling of one region: its coupled ports and the operations they need.

    A solver that supports coupling adds it in its init, with its mesh geometry,
    before it binds its coupled instances to their locations::

        ctx.models["coupling"] = Coupling(self.coupling, FoamGeometry(mesh))
        ...
        coupling(ctx).add(Patch("top"), bc)
    """

    def __init__(self, coupling: Optional[RegionCoupling], geometry: MeshGeometry) -> None:
        self.coupling = coupling
        self.geometry = geometry
        self._ports: dict[str, CoupledPort] = {}
        self._operations: Optional[Operations] = None

    @property
    def ports(self) -> tuple[CoupledPort, ...]:
        """The region's coupled ports, in the order they were added."""
        return tuple(self._ports.values())

    def add(self, location: Location, instance: Any) -> None:
        """Bind the ``@coupled`` ``instance`` to ``location``, before the graph is built.

        Instances on one location merge into its port.
        """
        declaration = getattr(type(instance), "__coupled__", None)
        if declaration is None:
            raise TypeError(f"{type(instance).__qualname__} is not @coupled")
        port = CoupledPort(
            self._region(location),
            location,
            declaration.streams(instance, "sends"),
            declaration.streams(instance, "receives"),
        )
        _check_entities(port)
        if self._operations is not None:
            raise ValueError(f"{port.label} is added after the region's graph was built")
        known = self._ports.get(port.name)
        self._ports[port.name] = port if known is None else known.merged(port)

    def _region(self, location: Location) -> str:
        coupling = self.coupling
        if coupling is None:
            raise ValueError(
                f"no exchange connects the port {location.name}: the case does not couple its "
                "region (system/regionCoupling.yaml: exchanges)"
            )
        if location.name not in coupling.ports:
            raise ValueError(
                f"no exchange connects the port {coupling.region}.{location.name}; a coupled "
                "instance needs its exchange (system/regionCoupling.yaml: exchanges)"
            )
        return coupling.region

    def operations(self) -> Operations:
        """The region's coupling ops, one per node, outer to inner; built once."""
        if self._operations is None:
            coupling = self.coupling
            self._operations = Operations() if coupling is None else self._build(coupling)
        return self._operations

    def _build(self, coupling: RegionCoupling) -> Operations:
        by_node = self._by_node(coupling)
        geometries = {port.name: port.location.geometry(self.geometry) for port in self.ports}
        ops: list[Operation] = []
        for node in coupling.nodes:
            couple = CouplingOp(coupling, node, by_node[node.name], geometries)
            metadata = OperationMetadata(
                op_name=couple.label, wraps=node.placement_of(coupling.region)
            )
            ops.append(Operation(func=couple, metadata=metadata))
        return Operations(ops)

    def _by_node(self, coupling: RegionCoupling) -> dict[str, list[CoupledPort]]:
        """The ports by node; ``ValueError`` when a data field is in two nodes."""
        by_node: dict[str, list[CoupledPort]] = {node.name: [] for node in coupling.nodes}
        in_node: dict[str, str] = {}
        for port in self.ports:
            name = coupling.node_of(port.name).name
            for data in sorted(port.names):
                if in_node.setdefault(data, name) != name:
                    raise ValueError(
                        f"the data {data} of region {coupling.region} is coupled in the nodes "
                        f"'{in_node[data]}' and '{name}' ({port.label}); a region's data field "
                        "is in at most one coupling node"
                    )
            by_node[name].append(port)
        return by_node

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
        names = operation_names(graph)
        missing = [name for name in (SET_TIME_STEP, INCREMENT_TIME) if name not in names]
        if missing:
            raise ValueError(
                f"region {coupling.region} steps in time coupled with {list(coupling.group)}, "
                f"but its graph has no {missing}; the coupled regions agree on the time step "
                f"between the solutionLoop's {SET_TIME_STEP} and {INCREMENT_TIME}"
            )
        return Operation(
            func=SequentialOp(TimeStep(coupling)),
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
        holder = serial_holder(graph, f"{coupling.region}.time_step", serial)
        if holder is not None:
            raise ValueError(
                f"coupling '{holder}' is serial but holds the time step of its regions; make it "
                f"parallel or place it around named operations "
                f"(wraps.{coupling.region}.operations)"
            )


def coupling(ctx: Context) -> Coupling:
    """The coupling model of the region ``ctx``; ``ValueError`` when its solver added none.

    Example::

        coupling(ctx).add(Patch("top"), bc)
    """
    model = ctx.models.get("coupling")
    if not isinstance(model, Coupling):
        raise ValueError(f"the solver of region {ctx.region} does not support coupling")
    return model


def _check_entities(port: CoupledPort) -> None:
    """Raise ``TypeError`` unless every model of ``port`` lives on its location's entities."""
    for stream in (*port.sends, *port.receives):
        if stream.model.entity not in port.location.entities:
            raise TypeError(
                f"{stream.model.__qualname__} is {base_name(stream.model)}, which "
                f"does not live on {port.location} ({port.label})"
            )
