# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Graph edits that a boundary condition or model registers on ``ctx.graph``.

The graphs are built as a solver's resolved time step: a ``time_loop`` around
``start_step`` (rotates old times, solves nothing), the operations that solve
fields, and ``write`` (stepped last, but depends on nothing). ``_RepeatSolution``
stands in for a boundary condition: built with the context, it wraps the
solution of its field in a loop of a fixed number of passes. Expected graphs are
nested ``(loop, [body])`` / name literals.
"""

from __future__ import annotations

from typing import Union

import pytest

from neofoam.framework.context import Context
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
    StepBuilder,
)
from neofoam.framework.types import OperationMetadata

Tree = list[Union[str, tuple[str, "Tree"]]]


class _Passes:
    """Loop predicate: ``count`` passes per run of the loop."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.done = 0

    def __call__(self, ctx: Context) -> bool:
        self.done += 1
        if self.done <= self.count:
            return True
        self.done = 0
        return False


class _RepeatSolution:
    """A test boundary condition that solves its field ``count`` times per time step."""

    def __init__(self, field: str, count: int, ctx: Context) -> None:
        self.loop = IterativeOp(_Passes(count))
        ctx.graph.wrap(ops=ctx.graph.solving(field), with_=self.loop)


def _op(
    name: str,
    log: list[str],
    depends_on: list[str] | None = None,
    before: list[str] | None = None,
    solves: list[str] | None = None,
) -> Operation:
    return Operation(
        func=SequentialOp(lambda ctx: log.append(name)),
        metadata=OperationMetadata(
            op_name=name, depends_on=depends_on, before=before, solves=solves or []
        ),
    )


def _time_step(log: list[str]) -> StepBuilder:
    """time_loop: start_step, solve_U, solve_T (solves T), update_rho, write."""
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(_Passes(1)), metadata=OperationMetadata(op_name="time_loop")
    )
    with builder.loop(time_loop) as time_builder:
        time_builder.step(_op("start_step", log))
        time_builder.step(_op("solve_U", log, depends_on=["start_step"], solves=["U"]))
        time_builder.step(_op("solve_T", log, depends_on=["start_step"], solves=["T"]))
        time_builder.step(_op("update_rho", log, depends_on=["solve_T"]))
        time_builder.step(_op("write", log))
    return DAGResolver().resolve(builder, Operations())


def _tree(ops: list[Operation]) -> Tree:
    return [
        (op.operation_name or "", _tree(op.sub_operations))
        if isinstance(op.func, IterativeOp)
        else op.operation_name or ""
        for op in ops
    ]


def test_graph_edits_without_edits_return_the_graph() -> None:
    graph = _time_step([])

    assert Context(fields={}, models={}).graph.apply(graph) is graph


def test_graph_edits_wrap_puts_the_solving_ops_in_a_loop() -> None:
    ctx = Context(fields={}, models={})
    _RepeatSolution("T", count=2, ctx=ctx)

    graph = ctx.graph.apply(_time_step([]))

    assert _tree(graph.operations.ops) == [
        (
            "time_loop",
            [
                "start_step",
                "solve_U",
                ("loop_solving_T", ["solve_T", "update_rho"]),
                "write",
            ],
        )
    ]


def test_graph_edits_wrap_repeats_the_loop_body() -> None:
    log: list[str] = []
    ctx = Context(fields={}, models={})
    _RepeatSolution("T", count=2, ctx=ctx)

    ctx.graph.apply(_time_step(log)).operations.run(ctx)

    assert log == [
        "start_step",
        "solve_U",
        "solve_T",
        "update_rho",
        "solve_T",
        "update_rho",
        "write",
    ]


def test_selector_picks_the_ops_downstream_of_the_solving_op() -> None:
    log: list[str] = []
    ops = [
        _op("start_step", log),
        _op("solve_T", log, depends_on=["start_step"], solves=["T"]),
        _op("flux", log, before=["update_rho"]),
        _op("heat_flux", log, depends_on=["solve_T"], before=["update_rho"]),
        _op("update_rho", log),
        _op("write", log),
    ]

    selected = Context(fields={}, models={}).graph.solving("T").select(ops)

    assert [op.operation_name for op in selected] == ["solve_T", "heat_flux", "update_rho"]


def test_graph_edits_wraps_sharing_a_loop_join_it() -> None:
    ctx = Context(fields={}, models={})
    loop = IterativeOp(_Passes(2))
    ctx.graph.wrap(ops=ctx.graph.solving("U"), with_=loop)
    ctx.graph.wrap(ops=ctx.graph.solving("T"), with_=loop)

    graph = ctx.graph.apply(_time_step([]))

    assert _tree(graph.operations.ops) == [
        (
            "time_loop",
            [
                "start_step",
                ("loop_solving_T_U", ["solve_U", "solve_T", "update_rho"]),
                "write",
            ],
        )
    ]


def test_graph_edits_second_wrap_of_a_field_wraps_the_first() -> None:
    ctx = Context(fields={}, models={})
    _RepeatSolution("T", count=2, ctx=ctx)
    _RepeatSolution("T", count=3, ctx=ctx)

    graph = ctx.graph.apply(_time_step([]))

    assert _tree(graph.operations.ops) == [
        (
            "time_loop",
            [
                "start_step",
                "solve_U",
                ("loop_solving_T", [("loop_solving_T", ["solve_T", "update_rho"])]),
                "write",
            ],
        )
    ]


def test_graph_edits_wrap_of_an_unsolved_field_raises() -> None:
    ctx = Context(fields={}, models={})
    _RepeatSolution("h", count=2, ctx=ctx)

    with pytest.raises(ValueError, match=r"no operation solves \['h'\]"):
        ctx.graph.apply(_time_step([]))
