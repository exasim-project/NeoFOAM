# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for scope extraction in DAG resolver."""

from foamadapter.framework.operations import (
    Operation,
    StepBuilder,
    SequentialOp,
    IterativeOp,
    DAGResolver,
)
from dag_test_helpers import (
    set_time_step,
    increment_time,
    time_loop_condition,
    initialize_residual,
    solve_iteration,
    inner_loop_condition,
)


def test_extract_flat_operations():
    """Test scope extraction from flat StepBuilder."""
    builder = StepBuilder()
    builder.step(Operation(func=SequentialOp(set_time_step), operation_name="op1"))
    builder.step(Operation(func=SequentialOp(increment_time), operation_name="op2"))

    resolver = DAGResolver()
    scopes = resolver._extract_scopes(builder)

    assert "root" in scopes
    assert len(scopes["root"]) == 2
    assert scopes["root"][0].operation_name == "op1"
    assert scopes["root"][1].operation_name == "op2"


def test_extract_nested_loop():
    """Test scope extraction with nested loops."""
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(time_loop_condition),
        operation_name="time_loop",
    )

    with builder.loop(time_loop) as tloop:
        tloop.step(
            Operation(
                func=SequentialOp(set_time_step),
                operation_name="set_dt",
            )
        )

        inner_loop = Operation(
            func=IterativeOp(inner_loop_condition),
            operation_name="inner_loop",
        )
        with tloop.loop(inner_loop) as iloop:
            iloop.step(
                Operation(
                    func=SequentialOp(initialize_residual),
                    operation_name="init_residual",
                )
            )
            iloop.step(
                Operation(
                    func=SequentialOp(solve_iteration),
                    operation_name="solve",
                )
            )

    resolver = DAGResolver()
    scopes = resolver._extract_scopes(builder)

    # Should have root, time_loop, and inner_loop scopes
    assert "root" in scopes
    assert "time_loop" in scopes
    assert "inner_loop" in scopes

    # Root should contain the time_loop operation
    assert len(scopes["root"]) == 1
    assert scopes["root"][0].operation_name == "time_loop"

    # time_loop should contain set_dt and inner_loop
    assert len(scopes["time_loop"]) == 2
    assert scopes["time_loop"][0].operation_name == "set_dt"
    assert scopes["time_loop"][1].operation_name == "inner_loop"

    # inner_loop should contain init_residual and solve
    assert len(scopes["inner_loop"]) == 2
    assert scopes["inner_loop"][0].operation_name == "init_residual"
    assert scopes["inner_loop"][1].operation_name == "solve"
