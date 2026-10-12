# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The coupling loop of two sides that wrap the solution of their fields.

``_Side`` stands in for a coupled boundary condition together with the operation
that solves its field: solving sets its value (one more each pass, or a fixed
one), it sends that value and logs what it receives. Both sides wrap their
operation with one ``CouplingOp``, so the resolved graph holds one loop.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest

from neofoam.framework.context import Context
from neofoam.framework.coupling import CoupledSide, CouplingOp, SideChange
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import Operation, Operations, SequentialOp, StepBuilder
from neofoam.framework.types import OperationMetadata


class _Side:
    """A test side of field ``name``: solving counts up (or sets ``fixed``), it sends the value."""

    def __init__(self, name: str, log: list[str], fixed: Optional[float] = None) -> None:
        self.name = name
        self.log = log
        self.fixed = fixed
        self.value = 0.0
        self.solves = 0

    def solve(self, ctx: Context) -> None:
        self.solves += 1
        self.value = self.fixed if self.fixed is not None else self.value + 1
        self.log.append(f"solve_{self.name}")

    def send(self) -> dict[str, Any]:
        return {"x": np.array([self.value])}

    def receive(self, x: Any) -> None:
        self.log.append(f"receive_{self.name} {x[0]}")

    def register(self, ctx: Context, coupling: CouplingOp) -> None:
        side = CoupledSide(send=self.send, receive=self.receive, coordinates=np.zeros((1, 1)))
        ctx.graph.wrap(ops=ctx.graph.solving(self.name), with_=coupling, side=side)

    def operation(self) -> Operation:
        return Operation(
            func=SequentialOp(self.solve),
            metadata=OperationMetadata(op_name=f"solve_{self.name}", solves=[self.name]),
        )


def _run(ctx: Context, *sides: _Side) -> None:
    builder = StepBuilder()
    for side in sides:
        builder.step(side.operation())
    ctx.graph.apply(DAGResolver().resolve(builder, Operations())).operations.run(ctx)


def _never(changes: list[SideChange]) -> bool:
    return False


def test_coupling_op_serial_side_receives_what_the_other_sent_in_the_same_pass() -> None:
    log: list[str] = []
    ctx = Context(fields={}, models={})
    coupling = CouplingOp(converged=_never, max_iterations=2)
    a, b = _Side("a", log), _Side("b", log)
    a.register(ctx, coupling)
    b.register(ctx, coupling)

    _run(ctx, a, b)

    assert log == [
        "receive_a 0.0",
        "solve_a",
        "receive_b 1.0",
        "solve_b",
        "receive_a 1.0",
        "solve_a",
        "receive_b 2.0",
        "solve_b",
    ]


def test_coupling_op_stops_once_the_sent_data_is_unchanged() -> None:
    ctx = Context(fields={}, models={})
    coupling = CouplingOp(max_iterations=10)
    a, b = _Side("a", [], fixed=5.0), _Side("b", [], fixed=7.0)
    a.register(ctx, coupling)
    b.register(ctx, coupling)

    _run(ctx, a, b)

    assert (a.solves, b.solves) == (2, 2)


def test_coupling_op_stops_at_max_iterations() -> None:
    ctx = Context(fields={}, models={})
    coupling = CouplingOp(converged=_never, max_iterations=3)
    a, b = _Side("a", []), _Side("b", [])
    a.register(ctx, coupling)
    b.register(ctx, coupling)

    _run(ctx, a, b)

    assert (a.solves, b.solves) == (3, 3)


def test_coupling_op_of_three_sides_raises() -> None:
    ctx = Context(fields={}, models={})
    coupling = CouplingOp()
    sides = [_Side(name, []) for name in ("a", "b", "c")]
    for side in sides:
        side.register(ctx, coupling)

    with pytest.raises(ValueError, match="a coupling joins two sides, this one has 3"):
        _run(ctx, *sides)


def test_coupling_op_shared_returns_the_coupling_a_driver_registered() -> None:
    ctx = Context(fields={}, models={})
    coupling = CouplingOp()
    ctx.graph.couplings["interface"] = coupling

    assert CouplingOp.shared(ctx, key=("T", "interface")) is coupling


def test_coupling_op_shared_gives_each_patch_its_own_coupling() -> None:
    ctx = Context(fields={}, models={})

    left = CouplingOp.shared(ctx, key=("T", "left"))

    assert CouplingOp.shared(ctx, key=("T", "left")) is left
    assert CouplingOp.shared(ctx, key=("T", "right")) is not left
