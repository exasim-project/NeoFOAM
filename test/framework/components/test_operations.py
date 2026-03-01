# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from neofoam.framework.context import Context
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    SequentialOp,
    StepBuilder,
)
from neofoam.framework.types import OperationMetadata, OperationNumber

from framework.conftest import MaxIterations
from framework.components.conftest import noop, make_seq_op


def test_step_builder() -> None:
    builder = StepBuilder()

    builder.step(make_seq_op("step1", 1))
    builder.step(make_seq_op("step2", 2))

    assert len(builder.operations) == 2

    builder.loop(
        Operation(
            func=SequentialOp(noop),
            metadata=OperationMetadata(
                op_name="loop1", operation_number=OperationNumber(3)
            ),
            sub_operations=[
                make_seq_op("loop1_step1", 4),
                make_seq_op("loop1_step2", 5),
            ],
        )
    )
    assert len(builder.operations) == 3
    assert len(builder.operations[-1].sub_operations) == 2


def test_builder_context() -> None:
    builder = StepBuilder()

    builder.step(make_seq_op("step1", 1))
    builder.step(make_seq_op("step2", 2))

    assert len(builder.operations) == 2

    loop = builder.loop(make_seq_op("loop1", 3))
    loop.step(make_seq_op("loop1_step1", 4))
    loop.step(make_seq_op("loop1_step2", 5))

    assert len(builder.operations) == 3
    assert len(builder.operations[-1].sub_operations) == 2


def test_builder_nested_context() -> None:
    builder = StepBuilder()

    builder.step(make_seq_op("step1", 1))
    builder.step(make_seq_op("step2", 2))

    assert len(builder.operations) == 2

    loop = builder.loop(make_seq_op("loop1", 3))
    loop.step(make_seq_op("loop1_step1", 4))
    loop.step(make_seq_op("loop1_step2", 5))

    assert len(loop.operations) == 2

    assert len(builder.operations) == 3
    assert len(builder.operations[-1].sub_operations) == 2

    builder.loop(make_seq_op("loop2", 6)).step(make_seq_op("loop2_step1", 7)).step(
        make_seq_op("loop2_step2", 8)
    )

    assert len(builder.operations) == 4
    assert len(builder.operations[-1].sub_operations) == 2


def test_operation_run() -> None:
    count = [0]

    def increment(ctx: Context) -> None:
        count[0] += 1

    opBuilder = StepBuilder()

    opBuilder.step(
        Operation(
            func=SequentialOp(increment),
            metadata=OperationMetadata(
                op_name="increment1", operation_number=OperationNumber(1)
            ),
        )
    )
    opBuilder.step(
        Operation(
            func=SequentialOp(increment),
            metadata=OperationMetadata(
                op_name="increment2", operation_number=OperationNumber(2)
            ),
        )
    )

    loop = opBuilder.loop(
        Operation(
            func=IterativeOp(MaxIterations(max_iters=4)),
            metadata=OperationMetadata(
                op_name="loop_increment", operation_number=OperationNumber(3)
            ),
        )
    )
    loop.step(
        Operation(
            func=SequentialOp(increment),
            metadata=OperationMetadata(
                op_name="looped_increment_1",
                operation_number=OperationNumber(4),
            ),
        )
    )
    loop.step(
        Operation(
            func=SequentialOp(increment),
            metadata=OperationMetadata(
                op_name="looped_increment_2",
                operation_number=OperationNumber(5),
            ),
        )
    )

    ops = opBuilder.operations
    ops.run(Context(fields={}, models={}, mesh={}))

    assert count[0] == 2 + 4 * 2  # 2 from sequential, 4 loops with 2 increments each


def test_step_builder_context_manager() -> None:
    """StepBuilder.loop() can be used as a context manager via with-statement."""
    builder = StepBuilder()

    builder.step(make_seq_op("step1", 1))

    with builder.loop(make_seq_op("loop1", 2)) as inner:
        inner.step(make_seq_op("loop1_step1", 3))
        inner.step(make_seq_op("loop1_step2", 4))

    assert len(builder.operations) == 2
    assert len(builder.operations[-1].sub_operations) == 2


def test_step_builder_nested_context_manager() -> None:
    """Context managers can be nested for multi-level loops."""
    builder = StepBuilder()

    builder.step(make_seq_op("step1", 1))

    with builder.loop(make_seq_op("outer_loop", 2)) as outer:
        outer.step(make_seq_op("outer_step", 3))
        with outer.loop(make_seq_op("inner_loop", 4)) as inner:
            inner.step(make_seq_op("inner_step", 5))

    assert len(builder.operations) == 2
    outer_loop = builder.operations[-1]
    assert len(outer_loop.sub_operations) == 2
    assert outer_loop.sub_operations[0].metadata.op_name == "outer_step"
    inner_loop = outer_loop.sub_operations[1]
    assert len(inner_loop.sub_operations) == 1
    assert inner_loop.sub_operations[0].metadata.op_name == "inner_step"


def test_step_builder_context_manager_runs_correctly() -> None:
    """Operations built with context manager execute correctly."""
    count = [0]

    def increment(ctx: Context) -> None:
        count[0] += 1

    builder = StepBuilder()

    builder.step(
        Operation(
            func=SequentialOp(increment),
            metadata=OperationMetadata(
                op_name="pre", operation_number=OperationNumber(1)
            ),
        )
    )

    with builder.loop(
        Operation(
            func=IterativeOp(MaxIterations(max_iters=3)),
            metadata=OperationMetadata(
                op_name="loop", operation_number=OperationNumber(2)
            ),
        )
    ) as inner:
        inner.step(
            Operation(
                func=SequentialOp(increment),
                metadata=OperationMetadata(
                    op_name="looped", operation_number=OperationNumber(3)
                ),
            )
        )

    builder.operations.run(Context(fields={}, models={}, mesh={}))
    assert count[0] == 1 + 3  # 1 pre + 3 loop iterations
