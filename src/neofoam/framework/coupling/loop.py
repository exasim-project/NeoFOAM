# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The coupling loop: repeats the solution of two coupled sides until their exchange converges.

A coupled boundary condition registers its side on ``ctx.graph``::

    ctx.graph.wrap(
        ops=ctx.graph.solving("T"),
        with_=CouplingOp.shared(ctx, key=("T", "interface")),
        side=CoupledSide(send=self.send, receive=self.receive, coordinates=points),
    )

When the graph edits are applied, every side gets a receive operation before the
operations of its field and a send operation after them, all inside one loop
driven by the :class:`CouplingOp`. The sides exchange in the order they are
stepped (serial): the second side receives what the first sent in the same pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np

from neofoam.framework.operations import IterativeOp, Operation, SequentialOp
from neofoam.framework.types import OperationMetadata

if TYPE_CHECKING:
    from neofoam.framework.context import Context

# What a side sends, by name: arrays with one entry per point of its patch.
Data = dict[str, Any]


@dataclass(frozen=True)
class CoupledSide:
    """One side of a coupled interface: what it sends, how it receives, where its points are.

    Use it as ``side=`` of :meth:`~neofoam.framework.graph.GraphEdits.wrap`.
    ``send()`` returns the side's data by name; ``receive(**data)`` takes the data
    of the other side, mapped to ``coordinates`` (one row per point of the patch).

    Example::

        CoupledSide(send=self.send, receive=self.receive, coordinates=patch.coordinates)
    """

    send: Callable[[], Data]
    receive: Callable[..., None]
    coordinates: Any


class SideChange(NamedTuple):
    """What one side sent in the previous and in the current pass, for a convergence check."""

    field: str
    previous: Data
    current: Data


# Hands the data one side sent to the other side: (data, source, target, iteration) -> data.
Transfer = Callable[[Data, CoupledSide, CoupledSide, int], Data]
# Whether the coupling has converged, from the change of every side's data in the last pass.
Convergence = Callable[[list[SideChange]], bool]


def identity_transfer(data: Data, source: CoupledSide, target: CoupledSide, iteration: int) -> Data:
    """Hand the data over unchanged: the sides share their points."""
    return data


def unchanged(changes: list[SideChange]) -> bool:
    """Converged once no side's data changes by more than round-off between two passes."""
    return all(
        np.allclose(change.current[name], change.previous[name], rtol=1e-12, atol=0.0)
        for change in changes
        for name in change.current
    )


@dataclass
class _Attached:
    """A side registered on a coupling, with the field whose solution it wraps."""

    field: str
    side: CoupledSide


class CouplingOp(IterativeOp):
    """The loop of a coupled interface: repeats both sides' solution until the exchange converges.

    Use :meth:`shared` from a coupled boundary condition, so both sides of an
    interface (possibly in two regions) wrap their solution with the same
    instance and join one loop. Before each pass it checks ``converged`` on what
    the sides sent in the last pass; it stops after ``max_iterations`` passes
    regardless. ``transfer`` maps (and may relax) the data of one side onto the
    other. A coupling joins exactly two sides. Not thread safe.

    Example::

        coupling = CouplingOp(converged=unchanged, max_iterations=50)
    """

    def __init__(
        self,
        converged: Convergence = unchanged,
        transfer: Transfer = identity_transfer,
        max_iterations: int = 100,
    ) -> None:
        super().__init__(self._next_pass)
        self.converged = converged
        self.transfer = transfer
        self.max_iterations = max_iterations
        self.sides: list[_Attached] = []
        # the pass of the current time step; 0 between time steps
        self.iteration = 0
        self._sent: list[Data] = []
        self._previous: list[Data] = []

    @classmethod
    def shared(cls, ctx: Context, key: tuple[str, str]) -> CouplingOp:
        """The coupling of the patch ``key = (field, patch)`` on ``ctx``.

        A driver that knows the interfaces fills ``ctx.graph.couplings`` before the
        boundary conditions are built, so both sides of an interface get the same
        instance; otherwise every patch gets its own.
        """
        _field, patch = key
        couplings = ctx.graph.couplings
        if patch not in couplings:
            couplings[patch] = cls()
        return couplings[patch]

    def attach(self, field: str, side: CoupledSide) -> None:
        """Register ``side``, which wraps the solution of ``field``."""
        self.sides.append(_Attached(field, side))

    def side_operations(self, side: CoupledSide, label: str) -> tuple[Operation, Operation]:
        """The receive and the send operation of an attached ``side``, named after ``label``."""
        index = next(i for i, attached in enumerate(self.sides) if attached.side is side)
        receive = Operation(
            func=SequentialOp(lambda _ctx: self._receive(index)),
            metadata=OperationMetadata(op_name=f"receive_{label}"),
        )
        send = Operation(
            func=SequentialOp(lambda _ctx: self._send(index)),
            metadata=OperationMetadata(op_name=f"send_{label}"),
        )
        return receive, send

    def _next_pass(self, ctx: Context) -> bool:
        if self.iteration == 0:
            if not self._sent:
                self._sent = [attached.side.send() for attached in self.sides]
            return self._start_pass()
        if self.iteration >= self.max_iterations or self.converged(self._changes()):
            self.iteration = 0
            return False
        return self._start_pass()

    def _start_pass(self) -> bool:
        self._previous = list(self._sent)
        self.iteration += 1
        return True

    def _changes(self) -> list[SideChange]:
        return [
            SideChange(attached.field, previous, current)
            for attached, previous, current in zip(self.sides, self._previous, self._sent)
        ]

    def _receive(self, index: int) -> None:
        partner = self._partner(index)
        data = self.transfer(
            self._sent[partner],
            self.sides[partner].side,
            self.sides[index].side,
            self.iteration,
        )
        self.sides[index].side.receive(**data)

    def _send(self, index: int) -> None:
        self._sent[index] = self.sides[index].side.send()

    def _partner(self, index: int) -> int:
        if len(self.sides) != 2:
            raise ValueError(
                f"a coupling joins two sides, this one has {len(self.sides)} "
                f"({[attached.field for attached in self.sides]})"
            )
        return 1 - index
