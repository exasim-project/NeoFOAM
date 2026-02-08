# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from foamadapter.framework.operations import (
    IterativeOp,
    Operation,
    SequentialOp,
    StepBuilder,
)


def function1():
    return 1


def test_step_builder():
    builder = StepBuilder()

    builder.step(Operation(func=function1, operation_name="step1", operation_number=1))
    builder.step(Operation(func=function1, operation_name="step2", operation_number=2))

    assert len(builder.operations) == 2

    builder.loop(
        Operation(
            func=function1,
            operation_name="loop1",
            operation_number=3,
            sub_operations=[
                Operation(
                    func=function1, operation_name="loop1_step1", operation_number=4
                ),
                Operation(
                    func=function1, operation_name="loop1_step2", operation_number=5
                ),
            ],
        )
    )
    assert len(builder.operations) == 3
    assert len(builder.operations[-1].sub_operations) == 2


def test_builder_context():
    builder = StepBuilder()

    with builder as steps:
        steps.step(
            Operation(func=function1, operation_name="step1", operation_number=1)
        )
        steps.step(
            Operation(func=function1, operation_name="step2", operation_number=2)
        )

    assert len(builder.operations) == 2

    with builder.loop(
        Operation(func=function1, operation_name="loop1", operation_number=3)
    ) as loop:
        loop.step(
            Operation(func=function1, operation_name="loop1_step1", operation_number=4)
        )
        loop.step(
            Operation(func=function1, operation_name="loop1_step2", operation_number=5)
        )

    assert len(builder.operations) == 3
    assert len(builder.operations[-1].sub_operations) == 2


def test_builder_nested_context():
    builder = StepBuilder()

    with builder as steps:
        steps.step(
            Operation(func=function1, operation_name="step1", operation_number=1)
        )
        steps.step(
            Operation(func=function1, operation_name="step2", operation_number=2)
        )

        assert len(builder.operations) == 2

        with builder.loop(
            Operation(func=function1, operation_name="loop1", operation_number=3)
        ) as loop:
            loop.step(
                Operation(
                    func=function1, operation_name="loop1_step1", operation_number=4
                )
            )
            loop.step(
                Operation(
                    func=function1, operation_name="loop1_step2", operation_number=5
                )
            )

            assert len(loop.operations) == 2

    assert len(builder.operations) == 3
    assert len(builder.operations[-1].sub_operations) == 2

    builder.loop(
        Operation(func=function1, operation_name="loop2", operation_number=6)
    ).step(
        Operation(func=function1, operation_name="loop2_step1", operation_number=7)
    ).step(Operation(func=function1, operation_name="loop2_step2", operation_number=8))

    assert len(builder.operations) == 4
    assert len(builder.operations[-1].sub_operations) == 2


class MaxIterations:
    def __init__(self, max_iters=5):
        self.max_iters = max_iters
        self.current_iter = 0

    def __call__(self, ctx):
        self.current_iter += 1
        if self.current_iter <= self.max_iters:
            return True
        return False


def test_operation_run():
    ran = {"count": 0}

    def increment(ctx):
        ctx["count"] += 1

    opBuilder = StepBuilder()

    with opBuilder as op:
        op.step(
            Operation(
                func=SequentialOp(increment),
                operation_name="increment1",
                operation_number=1,
            )
        )
        op.step(
            Operation(
                func=SequentialOp(increment),
                operation_name="increment2",
                operation_number=2,
            )
        )

        with op.loop(
            Operation(
                func=IterativeOp(MaxIterations(max_iters=4)),
                operation_name="loop_increment",
                operation_number=3,
            )
        ) as loop:
            loop.step(
                Operation(
                    func=SequentialOp(increment),
                    operation_name="looped_increment_1",
                    operation_number=4,
                )
            )
            loop.step(
                Operation(
                    func=SequentialOp(increment),
                    operation_name="looped_increment_2",
                    operation_number=5,
                )
            )

    ops = opBuilder.operations
    ops.run(ran)

    assert (
        ran["count"] == 2 + 4 * 2
    )  # 2 from sequential, 4 loops with 2 increments each
