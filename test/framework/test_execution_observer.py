# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The execution observer sees init steps, operations and loop tests in order."""

import threading
from typing import Any

from framework.conftest import MaxIterations
from neofoam.framework.context import Context
from neofoam.framework.execution_observer import ExecutionObserver, current, observe
from neofoam.framework.initialization.execution.executor import execute_lazy_inits
from neofoam.framework.initialization.init_step import InitStep
from neofoam.framework.operations import IterativeOp, Operation, SequentialOp, StepBuilder
from neofoam.framework.types import OperationMetadata


class Recorder(ExecutionObserver):
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def init_steps(self, steps: list[InitStep]) -> None:
        self.events.append(("init_steps", [s.name for s in steps]))

    def before_init_step(self, step: InitStep, context: dict[str, Any]) -> None:
        self.events.append(("before_init", step.name, sorted(context)))

    def after_init_step(self, step: InitStep, context: dict[str, Any], value: Any) -> None:
        self.events.append(("after_init", step.name, value))

    def before_operation(self, op: Operation, ctx: Context) -> None:
        self.events.append(("before", op.name))

    def after_operation(self, op: Operation, ctx: Context) -> None:
        self.events.append(("after", op.name))

    def before_loop_test(self, op: Operation, ctx: Context, iteration: int) -> None:
        self.events.append(("test", op.name, iteration))

    def after_loop_test(self, op: Operation, ctx: Context, iteration: int, result: bool) -> None:
        self.events.append(("tested", op.name, iteration, result))


def _seq(name: str) -> Operation:
    return Operation(func=SequentialOp(lambda ctx: None), metadata=OperationMetadata(op_name=name))


def _ctx() -> Context:
    return Context(fields={}, models={}, mesh={})


def test_operations_and_loop_tests_are_reported_in_order() -> None:
    builder = StepBuilder()
    with builder as b:
        b.step(_seq("first"))
        loop = Operation(
            func=IterativeOp(MaxIterations(max_iters=2)), metadata=OperationMetadata(op_name="loop")
        )
        with b.loop(loop) as body:
            body.step(_seq("inner"))

    recorder = Recorder()
    with observe(recorder):
        builder.operations.run(_ctx())

    assert recorder.events == [
        ("before", "first"),
        ("after", "first"),
        ("test", "loop", 1),
        ("tested", "loop", 1, True),
        ("before", "inner"),
        ("after", "inner"),
        ("test", "loop", 2),
        ("tested", "loop", 2, True),
        ("before", "inner"),
        ("after", "inner"),
        ("test", "loop", 3),
        ("tested", "loop", 3, False),
    ]


def test_init_steps_are_reported_with_order_and_values() -> None:
    steps = [
        InitStep(name="b", depends_on=["a"], initializer=lambda ctx: ctx["a"] + 1),
        InitStep(name="a", initializer=lambda ctx: 1),
    ]
    recorder = Recorder()
    with observe(recorder):
        execute_lazy_inits(steps)

    assert recorder.events == [
        ("init_steps", ["a", "b"]),
        ("before_init", "a", []),
        ("after_init", "a", 1),
        ("before_init", "b", ["a"]),
        ("after_init", "b", 2),
    ]


def test_observe_restores_previous_observer() -> None:
    outer, inner = Recorder(), Recorder()
    assert current() is None
    with observe(outer):
        with observe(inner):
            assert current() is inner
        assert current() is outer
    assert current() is None


def test_no_observer_runs_unchanged() -> None:
    count = [0]
    op = Operation(
        func=SequentialOp(lambda ctx: count.__setitem__(0, count[0] + 1)),
        metadata=OperationMetadata(op_name="count"),
    )
    op.run(_ctx())
    assert count == [1]


def test_observer_is_per_thread() -> None:
    """Overlapping scopes in two threads neither leak into nor undo each other."""
    first, second = Recorder(), Recorder()
    entered, release = threading.Event(), threading.Event()
    seen: dict[str, Any] = {}

    def other() -> None:
        with observe(second):
            entered.set()
            release.wait(5)
            seen["inside"] = current()
        seen["after"] = current()

    with observe(first):
        thread = threading.Thread(target=other)
        thread.start()
        entered.wait(5)
        assert current() is first
    release.set()  # the first scope ended while the second is still open
    thread.join(5)
    assert seen == {"inside": second, "after": None}
    assert current() is None


def test_init_steps_carry_the_region_they_initialize() -> None:
    steps = [InitStep(name="a", initializer=lambda ctx: 1)]
    recorder = Recorder()
    regions: list[Any] = []
    recorder.init_steps = lambda steps: regions.extend(s.region for s in steps)  # type: ignore[method-assign]

    with observe(recorder):
        execute_lazy_inits(steps, region="left")
        execute_lazy_inits([InitStep(name="b", initializer=lambda ctx: 1)])

    assert regions == ["left", None]
