# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Operations marked ``first``/``final`` run only in their wrapping loops' first/final pass.

Outside coupling loops they run every time; ``Context.run_every_pass`` runs a
final one in every pass.
"""

from __future__ import annotations

from typing import Any, Callable

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


def _op(name: str, run: Callable[[Context], object], **metadata: Any) -> Operation:
    return Operation(func=SequentialOp(run), metadata=OperationMetadata(op_name=name, **metadata))


def _logged(name: str, log: list[str], **metadata: Any) -> Operation:
    return _op(name, lambda ctx: log.append(name), **metadata)


def _loop(name: str, count: int, wraps: Placement | None = None, marks: bool = True) -> Operation:
    return Operation(
        func=IterativeOp(_Passes(count, marks)),
        metadata=OperationMetadata(op_name=name, wraps=wraps),
    )


def _context() -> Context:
    return Context(fields={}, models={})


def _resolve(builder: StepBuilder, *loops: Operation) -> Operations:
    return DAGResolver().resolve(builder, Operations(list(loops))).operations


def _first_step(log: list[str]) -> StepBuilder:
    """time_loop (2 steps): start_step (first), solve_T, write."""
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 2, marks=False)) as time_builder:
        time_builder.step(_logged("start_step", log, first=True))
        time_builder.step(_logged("solve_T", log, depends_on=["start_step"]))
        time_builder.step(_logged("write", log, depends_on=["solve_T"]))
    return builder


def _final_step(log: list[str]) -> StepBuilder:
    """time_loop (2 steps): start_step (first), solve_T, write (final)."""
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 2, marks=False)) as time_builder:
        time_builder.step(_logged("start_step", log, first=True))
        time_builder.step(_logged("solve_T", log, depends_on=["start_step"]))
        time_builder.step(_logged("write", log, final=True, depends_on=["solve_T"]))
    return builder


def test_a_running_loop_publishes_its_pass_on_ctx_loops() -> None:
    seen: list[list[LoopEntry]] = []
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 2, marks=False)) as time_builder:
        time_builder.step(_op("look", lambda ctx: seen.append(list(ctx.loops))))
    ops = _resolve(builder, _loop("couple", 2, Placement(), marks=False))
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
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
        time_builder.step(_op("fail", fail))
    ctx = _context()

    with pytest.raises(RuntimeError, match="boom"):
        builder.operations.run(ctx)

    assert ctx.loops == []


@pytest.mark.parametrize(
    ("passes", "expected"),
    [
        ((), ["start_step", "solve_T", "write"] * 2),
        ((("couple", 3),), ["start_step", *["solve_T", "write"] * 3] * 2),
        ((("outer", 2), ("inner", 2)), ["start_step", *["solve_T", "write"] * 4] * 2),
    ],
    ids=["outside_a_wrapping_loop", "in_a_wrapping_loop", "in_nested_wrapping_loops"],
)
def test_a_first_operation_runs_only_in_the_first_pass_of_its_wrapping_loops(
    passes: tuple[tuple[str, int], ...], expected: list[str]
) -> None:
    log: list[str] = []
    loops = [_loop(name, count, Placement()) for name, count in passes]

    _resolve(_first_step(log), *loops).run(_context())

    assert log == expected


def test_a_first_operation_in_an_inner_wrapping_loop_waits_for_the_outer_first_pass() -> None:
    """outer wraps solve_U and the inner loop, inner wraps start_T (first) and solve_T."""
    log: list[str] = []
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
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
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
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
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
        time_builder.step(_logged("prepare", log))
        time_builder.step(_logged("start_T", log, first=True, depends_on=["prepare"]))
        time_builder.step(_logged("solve_T", log, depends_on=["start_T"]))

    _resolve(builder, _loop("couple", 2, Placement(operations=("start_T",)))).run(_context())

    assert log == ["prepare", "start_T", "solve_T", "solve_T"]


def test_a_first_operation_after_a_first_one_resolves() -> None:
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
        time_builder.step(_logged("set_time_step", [], first=True))
        time_builder.step(_logged("increment_time", [], first=True, depends_on=["set_time_step"]))

    _resolve(builder, _loop("couple", 2, Placement()))


@pytest.mark.parametrize(
    ("mark", "marks", "expected"),
    [
        ("first", ({"first": True}, {}), [True, False]),
        ("final", ({}, {"final": True}), [False, True]),
    ],
)
def test_the_operation_decorators_set_the_pass_mark(
    mark: str, marks: tuple[dict[str, bool], dict[str, bool]], expected: list[bool]
) -> None:
    solver = SolverSpec(f"{mark}Solver")
    model = ModelSpec(f"{mark}Model")

    @solver.operation(**marks[0])
    def solver_one(self: object) -> None: ...

    @solver.operation(**marks[1])
    def solver_two(self: object) -> None: ...

    @model.operation(**marks[0])
    def model_one(self: object) -> None: ...

    @model.operation(**marks[1])
    def model_two(self: object) -> None: ...

    solver_ops = solver.instantiate().operations
    model_ops = ModelRuntime(spec=model, name=f"{mark}Model", config={}).operations
    assert [getattr(op.metadata, mark) for op in solver_ops] == expected
    assert [getattr(op.metadata, mark) for op in model_ops] == expected


def test_a_loop_entry_is_not_final_unless_its_test_marks_it() -> None:
    assert LoopEntry("loop", 0, False).final is False

    seen: list[LoopEntry] = []
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
        time_builder.step(_op("look", lambda ctx: seen.append(ctx.loops[-1])))
    ops = _resolve(builder, _loop("couple", 3, Placement()))

    ops.run(_context())

    assert seen == [LoopEntry("couple", i, True, i == 2) for i in range(3)]


@pytest.mark.parametrize(
    ("passes", "marks", "expected"),
    [
        ((), True, ["start_step", "solve_T", "write"] * 2),
        ((("couple", 3),), True, ["start_step", "solve_T", "solve_T", "solve_T", "write"] * 2),
        ((("couple", 1),), True, ["start_step", "solve_T", "write"] * 2),
        ((("couple", 2),), False, ["start_step", "solve_T", "solve_T"] * 2),
        # the inner loop's final pass of the outer loop's first pass skips write
        ((("outer", 2), ("inner", 2)), True, ["start_step", *["solve_T"] * 4, "write"] * 2),
    ],
    ids=[
        "outside_a_wrapping_loop",
        "in_a_wrapping_loop",
        "in_a_single_pass",
        "in_a_loop_that_marks_no_pass_final",
        "in_nested_wrapping_loops",
    ],
)
def test_a_final_operation_runs_only_in_the_final_pass_of_its_wrapping_loops(
    passes: tuple[tuple[str, int], ...], marks: bool, expected: list[str]
) -> None:
    log: list[str] = []
    loops = [_loop(name, count, Placement(), marks=marks) for name, count in passes]

    _resolve(_final_step(log), *loops).run(_context())

    assert log == expected


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

    _resolve(_final_step(log), _loop("couple", 3, Placement())).run(ctx)

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

    builder = _final_step(log)
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
    ops = _resolve(_final_step([]), _loop("couple", 2, Placement()))

    with pytest.raises(ValueError, match=message):
        ops.run(ctx)


def test_run_every_pass_naming_a_first_operation_while_running_raises() -> None:
    def add(ctx: Context) -> None:
        # e.g. a debugger UI, while the first time step runs
        ctx.run_every_pass.add("start_step")

    builder = _final_step([])
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
        time_builder.step(_logged("write", log, final=True))
        time_builder.step(_logged("report", log, depends_on=["write"]))

    _resolve(builder, _loop("couple", 2, Placement(operations=("solve_T",)))).run(_context())

    assert log == ["solve_T", "solve_T", "write", "report"]


def test_a_final_operation_after_a_final_one_resolves() -> None:
    log: list[str] = []
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", 1, marks=False)) as time_builder:
        time_builder.step(_logged("write", log, final=True))
        time_builder.step(_logged("report", log, final=True, depends_on=["write"]))

    _resolve(builder, _loop("couple", 2, Placement())).run(_context())

    assert log == ["write", "report"]


def test_metadata_marked_both_first_with_final_raises() -> None:
    with pytest.raises(ValueError, match=r"solve is marked first and final"):
        OperationMetadata(op_name="solve", first=True, final=True)


def test_a_model_operation_marked_both_first_with_final_raises() -> None:
    model = ModelSpec("bothModel")

    @model.operation(first=True, final=True)
    def solve(self: object) -> None: ...

    with pytest.raises(ValueError, match=r"solve is marked first and final"):
        ModelRuntime(spec=model, name="bothModel", config={}).operations
