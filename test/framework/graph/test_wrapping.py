# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The resolver places a contributed wrapping loop at the top or around named operations.

The graphs are a solver's time step: a ``time_loop`` around ``start_step``,
``solve_U``, ``solve_T`` (both after start_step), ``update_rho`` (after solve_T)
and ``write`` (stepped last, depends on nothing). A contributed loop is a model op
whose ``metadata.wraps`` is a :class:`Placement`. Expected graphs are nested
``(loop, [body])`` / name literals.
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
from neofoam.framework.types import OperationMetadata, OperationNumber, Placement

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


def _op(
    name: str,
    log: list[str],
    depends_on: list[str] | None = None,
    before: list[str] | None = None,
) -> Operation:
    return Operation(
        func=SequentialOp(lambda ctx: log.append(name)),
        metadata=OperationMetadata(op_name=name, depends_on=depends_on, before=before),
    )


def _loop(name: str, wraps: Placement | None = None, count: int = 1) -> Operation:
    return Operation(
        func=IterativeOp(_Passes(count)), metadata=OperationMetadata(op_name=name, wraps=wraps)
    )


def _around(*operations: str) -> Placement:
    return Placement(operations=operations)


def _time_step(log: list[str]) -> StepBuilder:
    """time_loop: start_step, solve_U, solve_T, update_rho, write."""
    builder = StepBuilder()
    with builder.loop(_loop("time_loop")) as time_builder:
        time_builder.step(_op("start_step", log))
        time_builder.step(_op("solve_U", log, depends_on=["start_step"]))
        time_builder.step(_op("solve_T", log, depends_on=["start_step"]))
        time_builder.step(_op("update_rho", log, depends_on=["solve_T"]))
        time_builder.step(_op("write", log))
    return builder


def _tree(ops: list[Operation]) -> Tree:
    return [
        (op.operation_name or "", _tree(op.sub_operations))
        if isinstance(op.func, IterativeOp)
        else op.operation_name or ""
        for op in ops
    ]


def _resolve(builder: StepBuilder, *loops: Operation) -> list[Operation]:
    return DAGResolver().resolve(builder, Operations(list(loops))).operations.ops


_BODY = ["start_step", "solve_U", "solve_T", "update_rho", "write"]


def test_a_loop_without_operations_wraps_the_whole_time_step() -> None:
    ops = _resolve(_time_step([]), _loop("couple", Placement()))

    assert _tree(ops) == [("time_loop", [("couple", _BODY)])]


def test_a_second_loop_without_operations_nests_inside_the_first() -> None:
    ops = _resolve(_time_step([]), _loop("outer", Placement()), _loop("inner", Placement()))

    assert _tree(ops) == [("time_loop", [("outer", [("inner", _BODY)])])]


def test_a_loop_without_operations_wraps_the_root_without_a_loop() -> None:
    builder = StepBuilder()
    builder.step(_op("solve_T", []))
    builder.step(_op("write", []))

    assert _tree(_resolve(builder, _loop("couple", Placement()))) == [
        ("couple", ["solve_T", "write"])
    ]


def test_a_loop_around_an_operation_holds_it_and_what_is_downstream() -> None:
    ops = _resolve(_time_step([]), _loop("couple", _around("solve_T")))

    assert _tree(ops) == [
        (
            "time_loop",
            ["start_step", "solve_U", ("couple", ["solve_T", "update_rho"]), "write"],
        )
    ]


def test_a_loop_around_an_operation_repeats_its_body() -> None:
    log: list[str] = []
    ops = _resolve(_time_step(log), _loop("couple", _around("solve_T"), count=2))

    Operations(ops).run(Context(fields={}, models={}))

    assert log == [
        "start_step",
        "solve_U",
        *["solve_T", "update_rho"] * 2,
        "write",
    ]


def test_a_loop_around_two_operations_takes_the_place_of_the_last() -> None:
    ops = _resolve(_time_step([]), _loop("couple", _around("solve_T", "solve_U")))

    assert _tree(ops) == [
        ("time_loop", ["start_step", ("couple", ["solve_U", "solve_T", "update_rho"]), "write"])
    ]


def test_a_before_target_is_downstream_too() -> None:
    builder = StepBuilder()
    builder.step(_op("solve_T", [], before=["limit_T"]))
    builder.step(_op("unrelated", []))
    builder.step(_op("limit_T", []))

    assert _tree(_resolve(builder, _loop("couple", _around("solve_T")))) == [
        "unrelated",
        ("couple", ["solve_T", "limit_T"]),
    ]


def test_a_loop_around_an_unknown_operation_raises_with_the_names() -> None:
    with pytest.raises(
        ValueError,
        match=r"couple wraps \['solve_h'\], which name no operation; the operations are "
        r"\['solve_T', 'solve_U', 'start_step', 'update_rho', 'write'\]",
    ):
        _resolve(_time_step([]), _loop("couple", _around("solve_T", "solve_h")))


def test_a_loop_around_the_outermost_loop_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"couple wraps \['time_loop'\], the outermost loop; it wraps operations or "
        r"loops inside it",
    ):
        _resolve(_time_step([]), _loop("couple", _around("time_loop")))


def _pimple_step(log: list[str], time_step: bool = False) -> StepBuilder:
    """time_loop: set_time_step, increment_time, inner_loop [solve_U, solve_p], update_T, write.

    With ``time_step`` a coupled region's time-step op runs between
    set_time_step and increment_time (numbered, as the coupling model adds it).
    """
    builder = StepBuilder()
    with builder.loop(_loop("time_loop")) as time_builder:
        time_builder.step(_op("set_time_step", log))
        time_builder.step(_op("increment_time", log, depends_on=["set_time_step"]))
        with time_builder.loop(_loop("inner_loop", count=2)) as inner_builder:
            inner_builder.step(_op("solve_U", log))
            inner_builder.step(_op("solve_p", log, depends_on=["solve_U"]))
        time_builder.step(_op("update_T", log, depends_on=["solve_p"]))
        time_builder.step(_op("write", log))
    return builder


def _time_step_op(log: list[str]) -> Operation:
    op = _op("fluid.time_step", log, depends_on=["set_time_step"], before=["increment_time"])
    op.metadata.operation_number = OperationNumber(0)
    return op


_INNER = ("inner_loop", ["solve_U", "solve_p"])


def test_a_loop_around_a_loop_holds_it_whole_and_what_is_downstream() -> None:
    """update_T depends on solve_p inside inner_loop: it is downstream of the loop."""
    ops = _resolve(_pimple_step([]), _loop("couple", _around("inner_loop")))

    assert _tree(ops) == [
        (
            "time_loop",
            ["set_time_step", "increment_time", ("couple", [_INNER, "update_T"]), "write"],
        )
    ]


def test_a_loop_around_a_loop_leaves_the_time_step_before_it() -> None:
    """The time-step op stays between set_time_step and increment_time, outside the loop."""
    ops = DAGResolver().resolve(
        _pimple_step([]),
        Operations([_time_step_op([]), _loop("couple", _around("inner_loop"))]),
    )

    assert _tree(ops.operations.ops) == [
        (
            "time_loop",
            [
                "set_time_step",
                "fluid.time_step",
                "increment_time",
                ("couple", [_INNER, "update_T"]),
                "write",
            ],
        )
    ]


def test_a_loop_around_a_loop_repeats_it_whole() -> None:
    """Each pass of the loop runs every pass of inner_loop."""
    log: list[str] = []
    ops = _resolve(_pimple_step(log), _loop("couple", _around("inner_loop"), count=2))

    Operations(ops).run(Context(fields={}, models={}))

    assert log == [
        "set_time_step",
        "increment_time",
        *[*["solve_U", "solve_p"] * 2, "update_T"] * 2,
        "write",
    ]


@pytest.mark.parametrize(
    ("inner", "expected"),
    [
        ("solve_p", [("inner_loop", ["solve_U", ("T.couple", ["solve_p"])]), "update_T"]),
        ("inner_loop", [("T.couple", [_INNER, "update_T"])]),
    ],
)
def test_a_loop_inside_a_loop_around_a_loop_nests_inside_it(inner: str, expected: Tree) -> None:
    ops = _resolve(
        _pimple_step([]),
        _loop("U.couple", _around("inner_loop")),
        _loop("T.couple", _around(inner)),
    )

    assert _tree(ops) == [
        (
            "time_loop",
            ["set_time_step", "increment_time", ("U.couple", expected), "write"],
        )
    ]


def test_a_loop_around_a_loop_that_holds_a_placed_loop_raises() -> None:
    """The loop around solve_p is listed first: the loop around inner_loop would hold it."""
    with pytest.raises(
        ValueError, match=r"U.couple would hold the coupling loop \['T.couple'\] placed before"
    ):
        _resolve(
            _pimple_step([]),
            _loop("T.couple", _around("solve_p")),
            _loop("U.couple", _around("inner_loop")),
        )


def test_a_loop_around_a_loop_and_an_operation_inside_it_raises() -> None:
    with pytest.raises(ValueError, match="lie in more than one loop scope; couple wraps"):
        _resolve(_pimple_step([]), _loop("couple", _around("inner_loop", "solve_p")))


def test_a_loop_around_an_unknown_name_lists_the_loops_inside_too() -> None:
    with pytest.raises(
        ValueError,
        match=r"the operations are \['increment_time', 'inner_loop', 'set_time_step', "
        r"'solve_U', 'solve_p', 'update_T', 'write'\]",
    ):
        _resolve(_pimple_step([]), _loop("couple", _around("solve_h")))


def test_a_loop_around_operations_of_several_scopes_raises() -> None:
    """solve_T runs inside a loop, solve_U outside: one loop cannot repeat both."""
    builder = StepBuilder()
    builder.step(_op("solve_U", []))
    with builder.loop(_loop("inner")) as inner:
        inner.step(_op("solve_T", []))

    with pytest.raises(ValueError, match="lie in more than one loop scope; couple wraps"):
        _resolve(builder, _loop("couple", _around("solve_T", "solve_U")))


def _coupled_step(log: list[str], edge: str) -> StepBuilder:
    """time_loop: start_step, solve_U, solve_T after solve_U (over ``edge``), update_rho, write."""
    builder = StepBuilder()
    with builder.loop(_loop("time_loop")) as time_builder:
        time_builder.step(_op("start_step", log))
        time_builder.step(
            _op(
                "solve_U",
                log,
                depends_on=["start_step"],
                before=["solve_T"] if edge == "before" else None,
            )
        )
        time_builder.step(
            _op(
                "solve_T",
                log,
                depends_on=["start_step", "solve_U"] if edge == "depends_on" else ["start_step"],
            )
        )
        time_builder.step(_op("update_rho", log, depends_on=["solve_T"]))
        time_builder.step(_op("write", log))
    return builder


@pytest.mark.parametrize("edge", ["depends_on", "before"])
def test_a_loop_around_an_operation_inside_a_placed_loop_nests_inside_it(edge: str) -> None:
    """solve_T runs after solve_U: the U loop holds the T loop."""
    ops = _resolve(
        _coupled_step([], edge),
        _loop("U.couple", _around("solve_U")),
        _loop("T.couple", _around("solve_T")),
    )

    assert _tree(ops) == [
        (
            "time_loop",
            [
                "start_step",
                ("U.couple", ["solve_U", ("T.couple", ["solve_T", "update_rho"])]),
                "write",
            ],
        )
    ]


@pytest.mark.parametrize("edge", ["depends_on", "before"])
def test_a_loop_that_would_hold_a_placed_loop_raises(edge: str) -> None:
    """The T loop is listed first: the U loop would have to hold it."""
    with pytest.raises(ValueError, match="list the outer coupling first"):
        _resolve(
            _coupled_step([], edge),
            _loop("T.couple", _around("solve_T")),
            _loop("U.couple", _around("solve_U")),
        )


def test_a_nested_loop_repeats_each_pass_of_the_outer_one() -> None:
    """Each pass of the U loop runs every pass of the T loop."""
    log: list[str] = []
    ops = _resolve(
        _coupled_step(log, "depends_on"),
        _loop("U.couple", _around("solve_U"), count=2),
        _loop("T.couple", _around("solve_T"), count=2),
    )

    Operations(ops).run(Context(fields={}, models={}))

    assert log == [
        "start_step",
        *["solve_U", *["solve_T", "update_rho"] * 2] * 2,
        "write",
    ]


def test_an_operation_upstream_only_runs_before_the_loop() -> None:
    """A does not depend on solve_T, so it runs before the loop; B depends on both."""
    builder = StepBuilder()
    with builder.loop(_loop("time_loop")) as time_builder:
        time_builder.step(_op("start_step", []))
        time_builder.step(_op("solve_T", [], depends_on=["start_step"]))
        time_builder.step(_op("A", [], depends_on=["start_step"]))
        time_builder.step(_op("B", [], depends_on=["solve_T", "A"]))
        time_builder.step(_op("write", []))

    ops = _resolve(builder, _loop("T.couple", _around("solve_T")))

    assert _tree(ops) == [
        ("time_loop", ["start_step", "A", ("T.couple", ["solve_T", "B"]), "write"])
    ]
