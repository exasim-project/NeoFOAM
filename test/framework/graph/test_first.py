# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Running loops on ``ctx.loops``, and operations that run once per step (``first``).

A loop op publishes its entry ``(name, iteration, first, wrapping)`` on
``ctx.loops`` while it runs, outermost first. An operation marked ``first`` runs
only while every wrapping loop around it is in its first pass: once per time step
inside coupling loops, every time outside of them.
"""

from __future__ import annotations

from typing import Callable

import pytest

from neofoam.framework.context import Context, LoopEntry
from neofoam.framework.graph import DAGResolver
from neofoam.framework.model import ModelRuntime, ModelSpec
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
    StepBuilder,
)
from neofoam.framework.solver.spec import SolverSpec
from neofoam.framework.types import OperationMetadata, Placement


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


def _op(
    name: str,
    run: Callable[[Context], object],
    first: bool = False,
    depends_on: list[str] | None = None,
    before: list[str] | None = None,
) -> Operation:
    return Operation(
        func=SequentialOp(run),
        metadata=OperationMetadata(op_name=name, first=first, depends_on=depends_on, before=before),
    )


def _logged(name: str, log: list[str], **metadata: object) -> Operation:
    return _op(name, lambda ctx: log.append(name), **metadata)  # type: ignore[arg-type]


def _loop(name: str, count: int, wraps: Placement | None = None) -> Operation:
    return Operation(
        func=IterativeOp(_Passes(count)), metadata=OperationMetadata(op_name=name, wraps=wraps)
    )


def _context() -> Context:
    return Context(fields={}, models={})


def _resolve(builder: StepBuilder, *loops: Operation) -> Operations:
    return DAGResolver().resolve(builder, Operations(list(loops))).operations


def test_a_running_loop_publishes_its_pass_on_ctx_loops() -> None:
    seen: list[list[LoopEntry]] = []
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 2)) as time_builder:
        time_builder.step(_op("look", lambda ctx: seen.append(list(ctx.loops))))
    ops = _resolve(builder, _loop("couple", 2, Placement()))
    ctx = _context()

    ops.run(ctx)

    time = [LoopEntry("time_loop", step, False) for step in (0, 1)]
    couple = [LoopEntry("couple", passes, True) for passes in (0, 1)]
    assert seen == [[step, passes] for step in time for passes in couple]
    assert ctx.loops == []


def test_a_loop_entry_holds_the_pass_its_loop_test_decides_on() -> None:
    seen: list[LoopEntry] = []

    def test(ctx: Context) -> bool:
        seen.append(ctx.loops[-1])
        return ctx.loops[-1].iteration < 2

    Operation(func=IterativeOp(test), metadata=OperationMetadata(op_name="loop")).run(_context())

    assert seen == [LoopEntry("loop", i, False) for i in (0, 1, 2)]
    assert [entry.first for entry in seen] == [True, False, False]


def test_a_failing_loop_leaves_ctx_loops_empty() -> None:
    def fail(ctx: Context) -> None:
        raise RuntimeError("boom")

    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1)) as time_builder:
        time_builder.step(_op("fail", fail))
    ctx = _context()

    with pytest.raises(RuntimeError, match="boom"):
        builder.operations.run(ctx)

    assert ctx.loops == []


def _time_step(log: list[str]) -> StepBuilder:
    """time_loop (2 steps): start_step (first), solve_T, write."""
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 2)) as time_builder:
        time_builder.step(_logged("start_step", log, first=True))
        time_builder.step(_logged("solve_T", log, depends_on=["start_step"]))
        time_builder.step(_logged("write", log, depends_on=["solve_T"]))
    return builder


def test_a_first_operation_outside_a_wrapping_loop_runs_every_time_step() -> None:
    log: list[str] = []

    _resolve(_time_step(log)).run(_context())

    assert log == ["start_step", "solve_T", "write"] * 2


def test_a_first_operation_in_a_wrapping_loop_runs_once_per_time_step() -> None:
    log: list[str] = []

    _resolve(_time_step(log), _loop("couple", 3, Placement())).run(_context())

    assert log == ["start_step", *["solve_T", "write"] * 3] * 2


def test_a_first_operation_in_nested_wrapping_loops_runs_in_their_first_pass_only() -> None:
    log: list[str] = []
    ops = _resolve(_time_step(log), _loop("outer", 2, Placement()), _loop("inner", 2, Placement()))

    ops.run(_context())

    assert log == ["start_step", *["solve_T", "write"] * 4] * 2


def test_a_first_operation_in_an_inner_wrapping_loop_waits_for_the_outer_first_pass() -> None:
    """outer wraps solve_U and the inner loop, inner wraps start_T (first) and solve_T."""
    log: list[str] = []
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1)) as time_builder:
        time_builder.step(_logged("solve_U", log))
        time_builder.step(_logged("start_T", log, first=True))
        time_builder.step(_logged("solve_T", log, depends_on=["start_T", "solve_U"]))
    outer = _loop("outer", 2, Placement(operations=("solve_U", "start_T")))
    inner = _loop("inner", 2, Placement(operations=("start_T",)))

    _resolve(builder, outer, inner).run(_context())

    # start_T runs once: in the first pass of the outer and of the inner loop
    assert log == ["solve_U", "start_T", "solve_T", "solve_T", "solve_U", "solve_T", "solve_T"]


@pytest.mark.parametrize("edge", ["depends_on", "before"])
def test_a_first_operation_after_one_of_every_pass_in_its_loop_raises(edge: str) -> None:
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1)) as time_builder:
        time_builder.step(_logged("solve_T", [], before=["update"] if edge == "before" else None))
        time_builder.step(
            _logged(
                "update", [], first=True, depends_on=["solve_T"] if edge == "depends_on" else None
            )
        )

    with pytest.raises(
        ValueError,
        match=r"update runs only in the first pass of the wrapping loop couple \(first\), "
        r"but depends on solve_T, which runs in every pass of it",
    ):
        _resolve(builder, _loop("couple", 2, Placement()))


def test_a_first_operation_after_one_outside_its_loop_resolves() -> None:
    log: list[str] = []
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1)) as time_builder:
        time_builder.step(_logged("prepare", log))
        time_builder.step(_logged("start_T", log, first=True, depends_on=["prepare"]))
        time_builder.step(_logged("solve_T", log, depends_on=["start_T"]))

    _resolve(builder, _loop("couple", 2, Placement(operations=("start_T",)))).run(_context())

    assert log == ["prepare", "start_T", "solve_T", "solve_T"]


def test_a_first_operation_after_a_first_one_resolves() -> None:
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1)) as time_builder:
        time_builder.step(_logged("set_time_step", [], first=True))
        time_builder.step(_logged("increment_time", [], first=True, depends_on=["set_time_step"]))

    _resolve(builder, _loop("couple", 2, Placement()))


def test_the_operation_decorators_mark_an_operation_first() -> None:
    solver = SolverSpec("firstSolver")
    model = ModelSpec("firstModel")

    @solver.operation(first=True)
    def start(self: object) -> None: ...

    @solver.operation()
    def solve(self: object) -> None: ...

    @model.operation(first=True)
    def start_model(self: object) -> None: ...

    @model.operation()
    def solve_model(self: object) -> None: ...

    solver_ops = solver.instantiate().operations
    model_ops = ModelRuntime(spec=model, name="firstModel", config={}).operations
    assert [op.metadata.first for op in solver_ops] == [True, False]
    assert [op.metadata.first for op in model_ops] == [True, False]
