# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Test regions for the coupling: a context per region with its coupling model, one thread each."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import cache, partial
from typing import Any, Callable, Optional, Sequence

import numpy as np
from pydantic import BaseModel, create_model

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Array,
    CommChannel,
    CouplingConvergence,
    CouplingLoop,
    Exchange,
    FaceData,
    InProcessCommChannel,
    Participant,
    ParticipantExchanges,
    Patch,
    StreamChange,
    coupled,
    coupling,
)
from neofoam.framework.coupling.exchanges import participant_exchanges
from neofoam.framework.coupling.iteration._leader import CouplingOutcome, Leader
from neofoam.framework.coupling.iteration._loop_op import CouplingLoopOp
from neofoam.framework.coupling.run import run_participants
from neofoam.framework.execution_observer import ExecutionObserver
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


class Outcomes(ExecutionObserver):
    """Records every coupling loop's outcome, one per time step, by its leader's label.

    The leader keeps only its last outcome; a test that checks the time steps observes
    the run with this (``with observe(outcomes): ...``). Thread safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_label: dict[str, list[CouplingOutcome]] = {}

    def coupling_finished(self, coupling: Leader, outcome: CouplingOutcome) -> None:
        with self._lock:
            self._by_label.setdefault(coupling.label, []).append(outcome)

    def of(self, label: str) -> tuple[CouplingOutcome, ...]:
        """The outcomes of the loop ``label`` (``<leader region>.<loop>``), in time order."""
        with self._lock:
            return tuple(self._by_label.get(label, ()))


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
    def __call__(self, changes: list[StreamChange]) -> bool:
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
    loops: Optional[Sequence[CouplingLoop]] = None,
    comm_channel: Optional[InProcessCommChannel] = None,
    **loop: Any,
) -> dict[str, Context]:
    """A context per region, coupled over ``exchanges`` in ``loops`` (default: one of ``loop``)."""
    if loops is None:
        loops = [CouplingLoop("coupling", tuple(exchange.name for exchange in exchanges), **loop)]
    couplings = participant_exchanges(
        regions, exchanges, comm_channel or InProcessCommChannel(), loops=loops
    )
    return {region: context(region, couplings.get(region)) for region in regions}


def context(region: str, region_coupling: Optional[ParticipantExchanges]) -> Context:
    """The context of ``region``, with its coupling model on a fake geometry."""
    return Context(
        fields={},
        models={"coupling": Participant(region_coupling, FakeGeometry())},
        region=region,
    )


@dataclass(frozen=True)
class CoupledInstance:
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


def fixed(
    value: float = 0.0, log: Optional[list[float]] = None, data: str = "v"
) -> CoupledInstance:
    """A coupled instance on one face that sends ``value`` as ``data`` and logs what it receives."""
    received = [] if log is None else log
    model = value_model(data)
    return CoupledInstance(
        model,
        model,
        lambda: model.model_validate({data: [value]}),
        lambda d: received.append(float(getattr(d, data)[0])),
        np.zeros((1, 1)),
    )


def add_coupled(ctx: Context, patch: str, instance: CoupledInstance) -> None:
    """Bind ``instance`` to ``Patch(patch)``, as a solver binds its ``@coupled`` instance."""

    def send(self: Any) -> Any:
        return instance.send()

    def receive(self: Any, data: Any) -> None:
        instance.receive(data)

    send.__annotations__ = {"return": instance.sends}
    receive.__annotations__ = {"data": instance.receives, "return": None}
    methods: dict[str, Any] = {}
    if instance.sends is not None:
        methods["send"] = coupled.sends(send)
    if instance.receives is not None:
        methods["receive"] = coupled.receives(receive)
    bound = coupled(type("CoupledInstance", (), methods))()
    if instance.coordinates is not None:
        coupling(ctx).geometry.centres[patch] = instance.coordinates  # type: ignore[attr-defined]
    coupling(ctx).add(Patch(patch), bound)


def couple(ctx: Context, index: int = 0) -> CouplingLoopOp:
    """The region's coupling op of its ``index`` th loop."""
    op = coupling(ctx).operations()[index].func
    assert isinstance(op, CouplingLoopOp)
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
    run_participants(
        {name: partial(graph(ctx, *ops).run, ctx) for name, (ctx, ops) in regions.items()},
        comm_channel(ctx),
    )


def comm_channel(ctx: Context) -> CommChannel:
    region = coupling(ctx).coupling
    assert region is not None
    return region.comm_channel
