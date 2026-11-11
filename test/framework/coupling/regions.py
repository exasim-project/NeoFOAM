# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Test regions for the coupling: a context per region with its coupling model, one thread each."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Sequence

import numpy as np
from pydantic import BaseModel

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Array,
    Coupling,
    CouplingConvergence,
    CouplingNode,
    InProcessTransport,
    Link,
    SideChange,
    Transport,
    coupled,
)
from neofoam.framework.coupling.loop import CouplingOp
from neofoam.framework.coupling.region import region_couplings, run_regions
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
    StepBuilder,
)
from neofoam.framework.types import OperationMetadata

# a receive that should raise at once times out after this instead
TIMEOUT = 30.0


class RecordingTransport(InProcessTransport):
    """An in-process transport that records every message it sends."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple[str, str, str, Any]] = []

    def send(self, source: str, target: str, tag: str, message: Any) -> None:
        self.sent.append((source, target, tag, message))
        super().send(source, target, tag, message)

    @property
    def tags(self) -> list[str]:
        return [tag for _source, _target, tag, _message in self.sent]


class _Never(CouplingConvergence):
    def __call__(self, changes: list[SideChange]) -> bool:
        return False


never = _Never()


def contexts(
    regions: list[str],
    links: list[Link],
    nodes: Optional[Sequence[CouplingNode]] = None,
    transport: Optional[InProcessTransport] = None,
    **node: Any,
) -> dict[str, Context]:
    """A context per region, coupled over ``links`` in ``nodes`` (default: one node of ``node``)."""
    if nodes is None:
        nodes = [CouplingNode("coupling", tuple(link.name for link in links), **node)]
    couplings = region_couplings(regions, links, transport or InProcessTransport(), nodes=nodes)
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


class Value(BaseModel):
    v: Array


def fixed(value: float = 0.0, log: Optional[list[float]] = None) -> Side:
    """A side of one point that sends ``value`` and appends what it receives to ``log``."""
    received = [] if log is None else log
    return Side(
        Value,
        Value,
        lambda: Value(v=[value]),
        lambda d: received.append(float(d.v[0])),
        np.zeros((1, 1)),
    )


def add_coupled(ctx: Context, patch: str, field: str, side: Side) -> None:
    """Add ``side`` on ``patch``, coupling ``field``, as a ``@coupled`` boundary condition does."""

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


def tree(ops: Sequence[Operation]) -> list[Any]:
    """The names of ``ops``, a loop as ``(name, tree of what it holds)``."""
    return [
        (op.operation_name, tree(op.sub_operations))
        if isinstance(op.func, IterativeOp)
        else op.operation_name
        for op in ops
    ]


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
    coupling = Coupling.of(ctx).coupling
    assert coupling is not None
    return coupling.transport
