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


# ---------------------------------------------------------------------------
# StepBuilder implicit chaining tests
# ---------------------------------------------------------------------------


def test_step_builder_chains_unnumbered_ops() -> None:
    """Unnumbered ops without explicit deps get chained by insertion order."""
    builder = StepBuilder()
    builder.step(make_seq_op("A"))
    builder.step(make_seq_op("B"))
    builder.step(make_seq_op("C"))

    ops = list(builder.operations)
    assert ops[0].depends_on == []
    assert ops[1].depends_on == ["A"]
    assert ops[2].depends_on == ["B"]


def test_step_builder_chains_numbered_ops() -> None:
    """Numbered ops also get chained by insertion order."""
    builder = StepBuilder()
    builder.step(make_seq_op("C", number="3"))
    builder.step(make_seq_op("A", number="1"))
    builder.step(make_seq_op("B", number="2"))

    ops = list(builder.operations)
    assert ops[0].depends_on == []
    assert ops[1].depends_on == ["C"]
    assert ops[2].depends_on == ["A"]


def test_step_builder_appends_to_explicit_deps() -> None:
    """Ops with explicit depends_on get the previous step appended."""
    builder = StepBuilder()
    builder.step(make_seq_op("A"))
    builder.step(make_seq_op("B", depends_on=["X"]))

    ops = list(builder.operations)
    assert ops[1].depends_on == ["X", "A"]


def test_step_builder_chains_loop() -> None:
    """Loops participate in the chain like any other op."""
    from framework.components.conftest import make_loop_op

    builder = StepBuilder()
    builder.step(make_seq_op("A"))
    builder.step(make_seq_op("B"))
    with builder.loop(make_loop_op("my_loop")) as inner:
        inner.step(make_seq_op("C"))
    builder.step(make_seq_op("D"))

    ops = list(builder.operations)
    assert ops[0].depends_on == []  # A
    assert ops[1].depends_on == ["A"]  # B
    assert ops[2].depends_on == ["B"]  # my_loop
    assert ops[3].depends_on == ["B"]  # D depends on B (last seq op before loop)


def test_step_builder_chains_nested_loops() -> None:
    """Nested loops set correct dependency chains at each nesting level.

    Mirrors the solver pattern:
        with builder.loop(time_loop) as time_builder:
            time_builder.step(set_time_step)
            time_builder.step(increment_time)
            with time_builder.loop(inner_loop) as inner_builder:
                inner_builder.step(momentum)
                inner_builder.step(continuity)
            time_builder.step(write_output)
    """
    from framework.components.conftest import make_loop_op

    builder = StepBuilder()
    builder.step(make_seq_op("init"))

    with builder.loop(make_loop_op("time_loop")) as time_builder:
        time_builder.step(make_seq_op("set_time_step"))
        time_builder.step(make_seq_op("increment_time"))

        with time_builder.loop(make_loop_op("inner_loop")) as inner_builder:
            inner_builder.step(make_seq_op("momentum"))
            inner_builder.step(make_seq_op("continuity"))

        time_builder.step(make_seq_op("write_output"))

    # -- root level: init -> time_loop
    root_ops = list(builder.operations)
    assert root_ops[0].depends_on == []  # init
    assert root_ops[1].depends_on == ["init"]  # time_loop

    # -- time_loop level: set_time_step -> increment_time -> inner_loop -> write_output
    time_ops = root_ops[1].sub_operations
    assert len(time_ops) == 4
    assert time_ops[0].depends_on == []  # set_time_step
    assert time_ops[1].depends_on == ["set_time_step"]  # increment_time
    assert time_ops[2].depends_on == ["increment_time"]  # inner_loop
    assert time_ops[3].depends_on == [
        "increment_time"
    ]  # write_output depends on last seq op before loop

    # -- inner_loop level: momentum -> continuity
    inner_ops = time_ops[2].sub_operations
    assert len(inner_ops) == 2
    assert inner_ops[0].depends_on == []  # momentum
    assert inner_ops[1].depends_on == ["momentum"]  # continuity
