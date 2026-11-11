# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Test regions for the coupling: a context per region with its coupling model, one thread each."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache, partial
from typing import Any, Callable, Optional, Sequence

import numpy as np
from pydantic import BaseModel, create_model

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Array,
    CommChannel,
    Coupling,
    CouplingConvergence,
    CouplingNode,
    Exchange,
    FaceData,
    InProcessCommChannel,
    Patch,
    RegionCoupling,
    SideChange,
    coupled,
    coupling,
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


class RecordingCommChannel(InProcessCommChannel):
    """An in-process comm channel that records every message it sends."""

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


@dataclass
class FakeGeometry:
    """A region's mesh geometry for the tests: face centres and areas by patch, the cells'."""

    centres: dict[str, Any] = field(default_factory=dict)
    areas: dict[str, Any] = field(default_factory=dict)
    cells: Any = None
    volumes: Any = None

    def face_centres(self, patch: str) -> Any:
        return self.centres.get(patch)

    def face_areas(self, patch: str) -> Any:
        return self.areas.get(patch)

    def cell_centres(self) -> Any:
        return self.cells

    def cell_volumes(self) -> Any:
        return self.volumes


def contexts(
    regions: list[str],
    exchanges: list[Exchange],
    nodes: Optional[Sequence[CouplingNode]] = None,
    comm_channel: Optional[InProcessCommChannel] = None,
    **node: Any,
) -> dict[str, Context]:
    """A context per region, coupled over ``exchanges`` in ``nodes`` (default: one of ``node``)."""
    if nodes is None:
        nodes = [CouplingNode("coupling", tuple(exchange.name for exchange in exchanges), **node)]
    couplings = region_couplings(
        regions, exchanges, comm_channel or InProcessCommChannel(), nodes=nodes
    )
    return {region: context(region, couplings.get(region)) for region in regions}


def context(region: str, region_coupling: Optional[RegionCoupling]) -> Context:
    """The context of ``region``, with its coupling model on a fake geometry."""
    return Context(
        fields={},
        models={"coupling": Coupling(region_coupling, FakeGeometry())},
        region=region,
    )


@dataclass(frozen=True)
class Side:
    """What a ``@coupled`` instance declares: its models (``None``: it has none) and methods.

    ``coordinates``: the face centres of its patch in the fake geometry.
    """

    sends: Optional[type[BaseModel]]
    receives: Optional[type[BaseModel]]
    send: Callable[[], BaseModel]
    receive: Callable[[Any], None]
    coordinates: Any = None


@cache
def value_model(data: str) -> type[FaceData]:
    """A FaceData model of the one Array field ``data``."""
    model: type[FaceData] = create_model(  # type: ignore[call-overload]
        f"Value_{data}", __base__=FaceData, **{data: (Array, ...)}
    )
    return model


def fixed(value: float = 0.0, log: Optional[list[float]] = None, data: str = "v") -> Side:
    """A side of one face that sends ``value`` as ``data`` and logs what it receives."""
    received = [] if log is None else log
    model = value_model(data)
    return Side(
        model,
        model,
        lambda: model.model_validate({data: [value]}),
        lambda d: received.append(float(getattr(d, data)[0])),
        np.zeros((1, 1)),
    )


def add_coupled(ctx: Context, patch: str, side: Side) -> None:
    """Bind ``side`` to ``Patch(patch)``, as a solver binds its ``@coupled`` instance."""

    def send(self: Any) -> Any:
        return side.send()

    def receive(self: Any, data: Any) -> None:
        side.receive(data)

    send.__annotations__ = {"return": side.sends}
    receive.__annotations__ = {"data": side.receives, "return": None}
    methods: dict[str, Any] = {}
    if side.sends is not None:
        methods["send"] = coupled.sends(send)
    if side.receives is not None:
        methods["receive"] = coupled.receives(receive)
    instance = coupled(type("Side", (), methods))()
    if side.coordinates is not None:
        coupling(ctx).geometry.centres[patch] = side.coordinates  # type: ignore[attr-defined]
    coupling(ctx).add(Patch(patch), instance)


def couple(ctx: Context, index: int = 0) -> CouplingOp:
    """The region's coupling op of its ``index`` th node."""
    op = coupling(ctx).operations()[index].func
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
    couple_ops = coupling(ctx).operations().ops
    return DAGResolver().resolve(StepBuilder(list(ops)), Operations(couple_ops)).operations


def run(regions: dict[str, tuple[Context, list[Operation]]]) -> None:
    """Run the graph of every region, each on its own thread."""
    (ctx, _ops), *_rest = regions.values()
    run_regions(
        {name: partial(graph(ctx, *ops).run, ctx) for name, (ctx, ops) in regions.items()},
        comm_channel(ctx),
    )


def comm_channel(ctx: Context) -> CommChannel:
    region = coupling(ctx).coupling
    assert region is not None
    return region.comm_channel
