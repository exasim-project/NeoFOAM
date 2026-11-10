# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Operations that run in the final pass of their wrapping loops (``final``).

A loop's test marks the pass it is about to run as the loop's last one
(``Context.mark_final``: ``LoopEntry.final``). An operation marked ``final`` runs
only while every wrapping loop around it is in its final pass: once per time step
inside coupling loops, every time outside of them. ``Context.run_every_pass``
runs one in every pass instead.
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
    """Loop test: ``count`` passes per run of the loop, the last one marked final."""

    def __init__(self, count: int, marks: bool = True) -> None:
        self.count = count
        self.marks = marks
        self.done = 0

    def __call__(self, ctx: Context) -> bool:
        self.done += 1
        if self.done <= self.count:
            if self.marks:
                ctx.mark_final(self.done == self.count)
            return True
        self.done = 0
        return False


def _op(
    name: str,
    run: Callable[[Context], object],
    first: bool = False,
    final: bool = False,
    depends_on: list[str] | None = None,
    before: list[str] | None = None,
) -> Operation:
    return Operation(
        func=SequentialOp(run),
        metadata=OperationMetadata(
            op_name=name, first=first, final=final, depends_on=depends_on, before=before
        ),
    )


def _logged(name: str, log: list[str], **metadata: object) -> Operation:
    return _op(name, lambda ctx: log.append(name), **metadata)  # type: ignore[arg-type]


def _loop(name: str, count: int, wraps: Placement | None = None, marks: bool = True) -> Operation:
    return Operation(
        func=IterativeOp(_Passes(count, marks)),
        metadata=OperationMetadata(op_name=name, wraps=wraps),
    )


def _context() -> Context:
    return Context(fields={}, models={})


def _resolve(builder: StepBuilder, *loops: Operation) -> Operations:
    return DAGResolver().resolve(builder, Operations(list(loops))).operations


def _time_step(log: list[str]) -> StepBuilder:
    """time_loop (2 steps): start_step (first), solve_T, write (final)."""
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 2, marks=False)) as time_builder:
        time_builder.step(_logged("start_step", log, first=True))
        time_builder.step(_logged("solve_T", log, depends_on=["start_step"]))
        time_builder.step(_logged("write", log, final=True, depends_on=["solve_T"]))
    return builder


def test_a_loop_entry_is_not_final_unless_its_test_marks_it() -> None:
    assert LoopEntry("loop", 0, False).final is False

    seen: list[LoopEntry] = []
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
        time_builder.step(_op("look", lambda ctx: seen.append(ctx.loops[-1])))
    ops = _resolve(builder, _loop("couple", 3, Placement()))

    ops.run(_context())

    assert seen == [LoopEntry("couple", i, True, i == 2) for i in range(3)]


def test_a_final_operation_outside_a_wrapping_loop_runs_every_time_step() -> None:
    log: list[str] = []

    _resolve(_time_step(log)).run(_context())

    assert log == ["start_step", "solve_T", "write"] * 2


def test_a_final_operation_in_a_wrapping_loop_runs_in_its_final_pass_only() -> None:
    log: list[str] = []

    _resolve(_time_step(log), _loop("couple", 3, Placement())).run(_context())

    assert log == ["start_step", "solve_T", "solve_T", "solve_T", "write"] * 2


def test_a_single_pass_is_first_and_final() -> None:
    log: list[str] = []

    _resolve(_time_step(log), _loop("couple", 1, Placement())).run(_context())

    assert log == ["start_step", "solve_T", "write"] * 2


def test_a_final_operation_in_a_wrapping_loop_that_marks_no_pass_final_never_runs() -> None:
    log: list[str] = []

    _resolve(_time_step(log), _loop("couple", 2, Placement(), marks=False)).run(_context())

    assert log == ["start_step", "solve_T", "solve_T"] * 2


def test_a_final_operation_in_nested_wrapping_loops_runs_in_their_final_passes_only() -> None:
    log: list[str] = []
    ops = _resolve(_time_step(log), _loop("outer", 2, Placement()), _loop("inner", 2, Placement()))

    ops.run(_context())

    # the inner loop's final pass of the outer loop's first pass skips write
    assert log == ["start_step", *["solve_T"] * 4, "write"] * 2


def test_a_final_operation_in_an_inner_final_pass_waits_for_the_outer_final_pass() -> None:
    """outer wraps solve_U and the inner loop, inner wraps solve_T and write_T (final)."""
    log: list[str] = []
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
        time_builder.step(_logged("solve_U", log))
        time_builder.step(_logged("solve_T", log, depends_on=["solve_U"]))
        time_builder.step(_logged("write_T", log, final=True, depends_on=["solve_T"]))
    outer = _loop("outer", 2, Placement(operations=("solve_U",)))
    inner = _loop("inner", 2, Placement(operations=("solve_T",)))

    _resolve(builder, outer, inner).run(_context())

    assert log == ["solve_U", "solve_T", "solve_T", "solve_U", "solve_T", "solve_T", "write_T"]


def test_run_every_pass_runs_a_final_operation_in_every_pass() -> None:
    log: list[str] = []
    ctx = _context()
    ctx.run_every_pass.add("write")

    _resolve(_time_step(log), _loop("couple", 3, Placement())).run(ctx)

    assert log == ["start_step", *["solve_T", "write"] * 3] * 2


def test_run_every_pass_taken_back_runs_it_in_the_final_pass_again() -> None:
    log: list[str] = []
    ctx = _context()
    ctx.run_every_pass.add("write")

    calls: list[None] = []

    def toggle(ctx: Context) -> None:
        # e.g. a debugger UI, at the start of the second pass
        calls.append(None)
        if len(calls) == 2:
            ctx.run_every_pass.discard("write")

    builder = _time_step(log)
    builder.operations[0].sub_operations.append(_op("toggle", toggle, before=["solve_T"]))
    _resolve(builder, _loop("couple", 2, Placement())).run(ctx)

    assert log == ["start_step", "solve_T", "write", "solve_T", "write"] + [
        "start_step",
        "solve_T",
        "solve_T",
        "write",
    ]


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("start_step", r"run_every_pass names start_step, which is first; only an operation "),
        ("solve_T", r"run_every_pass names solve_T, which is not marked final"),
        ("nope", r"run_every_pass names nope, which is no operation of the graph"),
    ],
)
def test_run_every_pass_naming_an_operation_not_marked_final_raises(
    name: str, message: str
) -> None:
    ctx = _context()
    ctx.run_every_pass.add(name)
    ops = _resolve(_time_step([]), _loop("couple", 2, Placement()))

    with pytest.raises(ValueError, match=message):
        ops.run(ctx)


def test_run_every_pass_naming_a_first_operation_while_running_raises() -> None:
    def add(ctx: Context) -> None:
        # e.g. a debugger UI, while the first time step runs
        ctx.run_every_pass.add("start_step")

    builder = _time_step([])
    builder.operations[0].sub_operations.append(_op("add", add, before=["solve_T"]))
    ops = _resolve(builder, _loop("couple", 2, Placement()))

    with pytest.raises(ValueError, match=r"run_every_pass names start_step, which is first"):
        ops.run(_context())


@pytest.mark.parametrize("edge", ["depends_on", "before"])
def test_an_operation_of_every_pass_after_a_final_one_in_its_loop_raises(edge: str) -> None:
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
        time_builder.step(
            _logged("write", [], final=True, before=["update"] if edge == "before" else None)
        )
        time_builder.step(
            _logged("update", [], depends_on=["write"] if edge == "depends_on" else None)
        )

    with pytest.raises(
        ValueError,
        match=r"update runs in every pass of the wrapping loop couple, but depends on write, "
        r"which runs only in its final pass \(final\)",
    ):
        _resolve(builder, _loop("couple", 2, Placement()))


def test_a_first_operation_after_a_final_one_in_its_loop_raises() -> None:
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
        time_builder.step(_logged("write", [], final=True))
        time_builder.step(_logged("start", [], first=True, depends_on=["write"]))

    with pytest.raises(ValueError, match=r"start runs in every pass .* depends on write"):
        _resolve(builder, _loop("couple", 2, Placement()))


def test_an_operation_after_a_final_one_outside_its_loop_resolves() -> None:
    log: list[str] = []
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
        time_builder.step(_logged("solve_T", log))
        time_builder.step(_logged("write", log, final=True, depends_on=["solve_T"]))
        time_builder.step(_logged("report", log, depends_on=["write"]))

    _resolve(builder, _loop("couple", 2, Placement(operations=("solve_T",), downstream=False))).run(
        _context()
    )

    assert log == ["solve_T", "solve_T", "write", "report"]


def test_a_final_operation_after_a_final_one_resolves() -> None:
    log: list[str] = []
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
        time_builder.step(_logged("write", log, final=True))
        time_builder.step(_logged("report", log, final=True, depends_on=["write"]))

    _resolve(builder, _loop("couple", 2, Placement())).run(_context())

    assert log == ["write", "report"]


def test_an_operation_marked_first_and_final_raises() -> None:
    with pytest.raises(ValueError, match=r"solve is marked first and final"):
        OperationMetadata(op_name="solve", first=True, final=True)

    model = ModelSpec("bothModel")

    @model.operation(first=True, final=True)
    def solve(self: object) -> None: ...

    with pytest.raises(ValueError, match=r"solve is marked first and final"):
        ModelRuntime(spec=model, name="bothModel", config={}).operations


def test_the_operation_decorators_mark_an_operation_final() -> None:
    solver = SolverSpec("finalSolver")
    model = ModelSpec("finalModel")

    @solver.operation()
    def solve(self: object) -> None: ...

    @solver.operation(final=True)
    def write(self: object) -> None: ...

    @model.operation()
    def solve_model(self: object) -> None: ...

    @model.operation(final=True)
    def write_model(self: object) -> None: ...

    solver_ops = solver.instantiate().operations
    model_ops = ModelRuntime(spec=model, name="finalModel", config={}).operations
    assert [op.metadata.final for op in solver_ops] == [False, True]
    assert [op.metadata.final for op in model_ops] == [False, True]
