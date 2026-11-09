# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Test regions for the coupling: a context per region, each run on its own thread.

Every region's context holds its coupling model (``ctx.models["coupling"]``), as
a solver that supports coupling adds it. A region's graph is one operation
``solve_<field>`` per field its sides couple, and the region's coupling ops (one
per coupling node), which the resolver wraps around them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, Optional, Sequence

from pydantic import BaseModel

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Coupling,
    CouplingNode,
    InProcessTransport,
    Link,
    Transport,
    coupled,
)
from neofoam.framework.coupling.loop import CouplingOp
from neofoam.framework.coupling.region import region_couplings, run_regions
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import Operation, Operations, SequentialOp, StepBuilder
from neofoam.framework.types import OperationMetadata


def contexts(
    regions: list[str],
    links: list[Link],
    scheme: Literal["serial", "parallel"] = "serial",
    nodes: Optional[Sequence[CouplingNode]] = None,
    **node: Any,
) -> dict[str, Context]:
    """A context per region, coupled over ``links`` in ``nodes``.

    Without ``nodes`` one node ``coupling`` of ``scheme`` holds every link;
    ``node`` sets its max_iterations, convergence and acceleration.
    """
    if nodes is None:
        names = tuple(link.name for link in links)
        nodes = [CouplingNode("coupling", names, scheme=scheme, **node)]
    couplings = region_couplings(regions, links, InProcessTransport(), nodes=nodes)
    return {
        region: Context(
            fields={},
            models={"coupling": Coupling(couplings.get(region))},
            region=region,
        )
        for region in regions
    }


@dataclass(frozen=True)
class Side:
    """What a ``@coupled`` boundary condition declares: its models, methods and points."""

    sends: type[BaseModel]
    receives: type[BaseModel]
    send: Callable[[], BaseModel]
    receive: Callable[[Any], None]
    coordinates: Any = None


def add_coupled(ctx: Context, patch: str, field: str, side: Side) -> None:
    """Add ``side`` on ``patch``, coupling ``field``, to the region's coupling model.

    As a ``@coupled`` boundary condition with ``side``'s models and methods does.
    """

    def send(self: Any) -> Any:
        return side.send()

    def receive(self: Any, data: Any) -> None:
        side.receive(data)

    def points(self: Any) -> Any:
        return side.coordinates

    send.__annotations__ = {"return": side.sends}
    receive.__annotations__ = {"data": side.receives, "return": None}
    cls = type(
        "Side",
        (),
        {
            "send": coupled.sends(send),
            "receive": coupled.receives(receive),
            "points": coupled.coordinates(points),
        },
    )
    instance = coupled(cls)()
    instance.patch, instance.field = patch, field
    Coupling.of(ctx).add(instance)


def couple(ctx: Context, index: int = 0) -> CouplingOp:
    """The region's coupling op of its ``index`` th node."""
    op = Coupling.of(ctx).operations()[index].func
    assert isinstance(op, CouplingOp)
    return op


def solve(field: str, func: Callable[[Context], None]) -> Operation:
    return Operation(
        func=SequentialOp(func),
        metadata=OperationMetadata(op_name=f"solve_{field}"),
    )


def graph(ctx: Context, *ops: Operation) -> Operations:
    """The resolved graph of the region ``ctx``: ``ops`` and its coupling ops."""
    couple_ops = Coupling.of(ctx).operations().ops
    return DAGResolver().resolve(StepBuilder(list(ops)), Operations(couple_ops)).operations


def run(regions: dict[str, tuple[Context, list[Operation]]]) -> None:
    """Run the graph of every region, each on its own thread."""
    (ctx, _ops), *_rest = regions.values()
    run_regions(
        {name: partial(graph(ctx, *ops).run, ctx) for name, (ctx, ops) in regions.items()},
        transport(ctx),
    )


def transport(ctx: Context) -> Transport:
    """The transport of the coupled region ``ctx``."""
    coupling = Coupling.of(ctx).coupling
    assert coupling is not None
    return coupling.transport
