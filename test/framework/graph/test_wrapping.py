# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The resolver nests the solution of the fields a contributed loop wraps inside it.

The graphs are a solver's time step: a ``time_loop`` around ``start_step``
(solves nothing), ``solve_U``, ``solve_T`` (solves T), ``update_rho`` (after
solve_T) and ``write`` (stepped last, depends on nothing). A contributed loop is
a model op whose ``metadata.wraps`` names fields. Expected graphs are nested
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


def _loop(name: str, wraps: list[str], count: int = 1) -> Operation:
    return Operation(
        func=IterativeOp(_Passes(count)), metadata=OperationMetadata(op_name=name, wraps=wraps)
    )


def _time_step(log: list[str]) -> StepBuilder:
    """time_loop: start_step, solve_U, solve_T (solves T), update_rho, write."""
    builder = StepBuilder()
    with builder.loop(_loop("time_loop", [])) as time_builder:
        time_builder.step(_op("start_step", log))
        time_builder.step(_op("solve_U", log, depends_on=["start_step"], solves=["U"]))
        time_builder.step(_op("solve_T", log, depends_on=["start_step"], solves=["T"]))
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


def test_a_loop_wrapping_a_field_holds_its_solution_and_what_is_downstream() -> None:
    ops = _resolve(_time_step([]), _loop("couple", ["T"]))

    assert _tree(ops) == [
        (
            "time_loop",
            ["start_step", "solve_U", ("couple", ["solve_T", "update_rho"]), "write"],
        )
    ]
    assert ops[0].sub_operations[2].metadata.solves == ["T"]


def test_a_loop_wrapping_a_field_repeats_its_body() -> None:
    log: list[str] = []
    ops = _resolve(_time_step(log), _loop("couple", ["T"], count=2))

    Operations(ops).run(Context(fields={}, models={}))

    assert log == [
        "start_step",
        "solve_U",
        *["solve_T", "update_rho"] * 2,
        "write",
    ]


def test_a_loop_wrapping_two_fields_takes_the_place_of_the_last() -> None:
    ops = _resolve(_time_step([]), _loop("couple", ["T", "U"]))

    assert _tree(ops) == [
        ("time_loop", ["start_step", ("couple", ["solve_U", "solve_T", "update_rho"]), "write"])
    ]


def test_a_before_target_is_downstream_too() -> None:
    builder = StepBuilder()
    builder.step(_op("solve_T", [], solves=["T"], before=["limit_T"]))
    builder.step(_op("unrelated", []))
    builder.step(_op("limit_T", []))

    assert _tree(_resolve(builder, _loop("couple", ["T"]))) == [
        "unrelated",
        ("couple", ["solve_T", "limit_T"]),
    ]


def test_a_second_loop_of_a_field_wraps_the_first() -> None:
    ops = _resolve(_time_step([]), _loop("inner", ["T"]), _loop("outer", ["T"]))

    assert _tree(ops) == [
        (
            "time_loop",
            [
                "start_step",
                "solve_U",
                ("outer", [("inner", ["solve_T", "update_rho"])]),
                "write",
            ],
        )
    ]


def test_a_loop_wrapping_an_unsolved_field_raises() -> None:
    with pytest.raises(ValueError, match=r"couple wraps \['h'\], which no operation solves"):
        _resolve(_time_step([]), _loop("couple", ["T", "h"]))


def test_a_loop_wrapping_fields_of_several_scopes_raises() -> None:
    """T is solved inside a loop, U outside: one loop cannot repeat both."""
    builder = StepBuilder()
    builder.step(_op("solve_U", [], solves=["U"]))
    with builder.loop(_loop("inner", [])) as inner:
        inner.step(_op("solve_T", [], solves=["T"]))

    with pytest.raises(ValueError, match="lie in more than one loop scope; couple wraps"):
        _resolve(builder, _loop("couple", ["T", "U"]))
