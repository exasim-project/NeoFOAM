# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from neofoam import telemetry
from neofoam.framework.context import Context
from neofoam.framework.operations import (
    ConditionalOp,
    IterativeOp,
    Operation,
    SequentialOp,
    StepBuilder,
)
from neofoam.framework.types import OperationMetadata, OperationNumber
from neofoam.telemetry import MpiInfo, TelemetrySettings

from framework.conftest import MaxIterations

HAS_OTEL = importlib.util.find_spec("opentelemetry") is not None


def function1(_ctx: Context) -> int:
    return 1


def _op(name: str, number: int, **kw: Any) -> Operation:
    """Shorthand for Operation with metadata."""
    return Operation(
        func=SequentialOp(function1),
        metadata=OperationMetadata(
            op_name=name, operation_number=OperationNumber(number), **kw
        ),
    )


def test_step_builder() -> None:
    builder = StepBuilder()

    builder.step(_op("step1", 1))
    builder.step(_op("step2", 2))

    assert len(builder.operations) == 2

    builder.loop(
        Operation(
            func=SequentialOp(function1),
            metadata=OperationMetadata(
                op_name="loop1", operation_number=OperationNumber(3)
            ),
            sub_operations=[
                _op("loop1_step1", 4),
                _op("loop1_step2", 5),
            ],
        )
    )
    assert len(builder.operations) == 3
    assert len(builder.operations[-1].sub_operations) == 2


def test_builder_context() -> None:
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


def test_builder_nested_context() -> None:
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


def test_operation_run() -> None:
    count = [0]

    def increment(ctx: Context) -> None:
        count[0] += 1

    opBuilder = StepBuilder()

    with opBuilder as op:
        op.step(
            Operation(
                func=SequentialOp(increment),
                metadata=OperationMetadata(
                    op_name="increment1", operation_number=OperationNumber(1)
                ),
            )
        )
        op.step(
            Operation(
                func=SequentialOp(increment),
                metadata=OperationMetadata(
                    op_name="increment2", operation_number=OperationNumber(2)
                ),
            )
        )

        with op.loop(
            Operation(
                func=IterativeOp(MaxIterations(max_iters=4)),
                metadata=OperationMetadata(
                    op_name="loop_increment", operation_number=OperationNumber(3)
                ),
            )
        ) as loop:
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


# --- telemetry instrumentation ------------------------------------------------


@pytest.fixture
def traced(tmp_path: Path) -> Iterator[Path]:
    """Activate telemetry for one test; yields the case dir."""
    telemetry.configure(TelemetrySettings(), case_dir=tmp_path, mpi=MpiInfo())
    yield tmp_path
    telemetry.shutdown()


def _span_records(case_dir: Path) -> list[dict[str, Any]]:
    telemetry.shutdown()  # flush
    path = case_dir / "telemetry" / "rank0.spans.jsonl"
    assert path.is_file(), f"missing span file {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _loop_graph(count: list[int]) -> StepBuilder:
    """iterative 'outer_loop' (2 iterations) wrapping sequential 'body'."""

    def body(_ctx: Context) -> None:
        count[0] += 1

    builder = StepBuilder()
    with builder.loop(
        Operation(
            func=IterativeOp(MaxIterations(max_iters=2)),
            metadata=OperationMetadata(op_name="outer_loop"),
        )
    ) as loop:
        loop.step(
            Operation(
                func=SequentialOp(body),
                metadata=OperationMetadata(op_name="body"),
            )
        )
    return builder


def test_run_without_telemetry_creates_no_spans(tmp_path: Path) -> None:
    count = [0]
    _loop_graph(count).operations.run(Context(fields={}, models={}, mesh={}))
    assert count[0] == 2
    assert not (tmp_path / "telemetry").exists()


def test_conditional_run_return_value_preserved_when_inactive() -> None:
    op = Operation(
        func=ConditionalOp(lambda _ctx: True),
        metadata=OperationMetadata(op_name="predicate"),
    )
    assert op.run(Context(fields={}, models={}, mesh={})) is True


@pytest.mark.skipif(not HAS_OTEL, reason="requires the neofoam[telemetry] extra")
def test_operation_spans_nest_under_loop_span(traced: Path) -> None:
    count = [0]
    _loop_graph(count).operations.run(Context(fields={}, models={}, mesh={}))
    assert count[0] == 2

    records = _span_records(traced)
    loops = [r for r in records if r["name"] == "outer_loop"]
    bodies = [r for r in records if r["name"] == "body"]

    assert len(loops) == 1
    assert len(bodies) == 2  # one span per loop iteration
    (loop,) = loops
    assert all(b["parent_id"] == loop["context"]["span_id"] for b in bodies)
    assert loop["attributes"]["operation_type"] == "iterative"
    assert bodies[0]["attributes"]["operation_type"] == "sequential"


@pytest.mark.skipif(not HAS_OTEL, reason="requires the neofoam[telemetry] extra")
def test_conditional_run_return_value_preserved_when_active(traced: Path) -> None:
    op = Operation(
        func=ConditionalOp(lambda _ctx: True),
        metadata=OperationMetadata(op_name="predicate"),
    )
    assert op.run(Context(fields={}, models={}, mesh={})) is True


@pytest.mark.skipif(not HAS_OTEL, reason="requires the neofoam[telemetry] extra")
def test_user_span_inside_operation_nests_under_operation_span(traced: Path) -> None:
    def solve(_ctx: Context) -> None:
        with telemetry.span("user.assemble"):
            pass

    op = Operation(
        func=SequentialOp(solve), metadata=OperationMetadata(op_name="momentum")
    )
    op.run(Context(fields={}, models={}, mesh={}))

    records = {r["name"]: r for r in _span_records(traced)}
    momentum = records["momentum"]
    assert records["user.assemble"]["parent_id"] == momentum["context"]["span_id"]


@pytest.mark.skipif(not HAS_OTEL, reason="requires the neofoam[telemetry] extra")
def test_unnamed_operation_gets_fallback_span_name(traced: Path) -> None:
    op = Operation(func=SequentialOp(lambda _ctx: None))
    op.run(Context(fields={}, models={}, mesh={}))

    names = {r["name"] for r in _span_records(traced)}
    assert names == {"operation"}
