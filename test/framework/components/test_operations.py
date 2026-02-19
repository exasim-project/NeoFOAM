# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    SequentialOp,
    StepBuilder,
)
from neofoam.framework.types import OperationMetadata

from conftest import MaxIterations


def function1():
    return 1


def _op(name, number, **kw):
    """Shorthand for Operation with metadata."""
    return Operation(
        func=function1,
        metadata=OperationMetadata(op_name=name, operation_number=number, **kw),
    )


def test_step_builder():
    builder = StepBuilder()

    builder.step(_op("step1", 1))
    builder.step(_op("step2", 2))

    assert len(builder.operations) == 2

    builder.loop(
        Operation(
            func=function1,
            metadata=OperationMetadata(op_name="loop1", operation_number=3),
            sub_operations=[
                _op("loop1_step1", 4),
                _op("loop1_step2", 5),
            ],
        )
    )
    assert len(builder.operations) == 3
    assert len(builder.operations[-1].sub_operations) == 2


def test_builder_context():
    builder = StepBuilder()

    with builder as steps:
        steps.step(_op("step1", 1))
        steps.step(_op("step2", 2))

    assert len(builder.operations) == 2

    with builder.loop(_op("loop1", 3)) as loop:
        loop.step(_op("loop1_step1", 4))
        loop.step(_op("loop1_step2", 5))

    assert len(builder.operations) == 3
    assert len(builder.operations[-1].sub_operations) == 2


def test_builder_nested_context():
    builder = StepBuilder()

    with builder as steps:
        steps.step(_op("step1", 1))
        steps.step(_op("step2", 2))

        assert len(builder.operations) == 2

        with builder.loop(_op("loop1", 3)) as loop:
            loop.step(_op("loop1_step1", 4))
            loop.step(_op("loop1_step2", 5))

            assert len(loop.operations) == 2

    assert len(builder.operations) == 3
    assert len(builder.operations[-1].sub_operations) == 2

    builder.loop(_op("loop2", 6)).step(_op("loop2_step1", 7)).step(
        _op("loop2_step2", 8)
    )

    assert len(builder.operations) == 4
    assert len(builder.operations[-1].sub_operations) == 2


def test_operation_run():
    ran = {"count": 0}

    def increment(ctx):
        ctx["count"] += 1

    opBuilder = StepBuilder()

    with opBuilder as op:
        op.step(
            Operation(
                func=SequentialOp(increment),
                metadata=OperationMetadata(op_name="increment1", operation_number=1),
            )
        )
        op.step(
            Operation(
                func=SequentialOp(increment),
                metadata=OperationMetadata(op_name="increment2", operation_number=2),
            )
        )

        with op.loop(
            Operation(
                func=IterativeOp(MaxIterations(max_iters=4)),
                metadata=OperationMetadata(
                    op_name="loop_increment", operation_number=3
                ),
            )
        ) as loop:
            loop.step(
                Operation(
                    func=SequentialOp(increment),
                    metadata=OperationMetadata(
                        op_name="looped_increment_1", operation_number=4
                    ),
                )
            )
            loop.step(
                Operation(
                    func=SequentialOp(increment),
                    metadata=OperationMetadata(
                        op_name="looped_increment_2", operation_number=5
                    ),
                )
            )

    ops = opBuilder.operations
    ops.run(ran)

    assert (
        ran["count"] == 2 + 4 * 2
    )  # 2 from sequential, 4 loops with 2 increments each
