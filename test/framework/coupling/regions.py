# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Test regions for the coupling: a context per region, each run on its own thread.

Every region's context holds its coupling model (``ctx.models["coupling"]``), as
a solver that supports coupling adds it. A region's graph is one operation
``solve_<field>`` per field its sides couple, and the region's coupling op,
which the resolver wraps around them.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    CoupledSide,
    Coupling,
    CouplingController,
    CouplingOp,
    InProcessTransport,
    Link,
    Transport,
    coupled,
    region_couplings,
    run_regions,
)
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import Operation, Operations, SequentialOp, StepBuilder
from neofoam.framework.types import OperationMetadata


def contexts(
    regions: list[str], links: list[Link], parallel: bool = False, **controller: Any
) -> dict[str, Context]:
    """A context per region, coupled over ``links``; ``controller`` sets the leader's controller."""
    couplings = region_couplings(
        regions,
        links,
        InProcessTransport(),
        parallel=parallel,
        controller=partial(CouplingController, **controller),
    )
    return {
        region: Context(
            fields={},
            models={"coupling": Coupling(couplings.get(region))},
            region=region,
        )
        for region in regions
    }


def add_coupled(ctx: Context, patch: str, field: str, side: CoupledSide) -> None:
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


def couple(ctx: Context) -> CouplingOp:
    """The region's coupling op."""
    op = Coupling.of(ctx).operations()[0].func
    assert isinstance(op, CouplingOp)
    return op


def solve(field: str, func: Callable[[Context], None]) -> Operation:
    return Operation(
        func=SequentialOp(func),
        metadata=OperationMetadata(op_name=f"solve_{field}"),
    )


def graph(ctx: Context, *ops: Operation) -> Operations:
    """The resolved graph of the region ``ctx``: ``ops`` and its coupling op."""
    couple_op = Coupling.of(ctx).operations().ops[:1]
    return DAGResolver().resolve(StepBuilder(list(ops)), Operations(couple_op)).operations


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
