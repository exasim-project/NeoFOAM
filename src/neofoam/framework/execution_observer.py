# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Hooks through which a tool watches — or pauses — a solver run.

The framework calls the installed :class:`ExecutionObserver` around every init
step, every operation and every loop-predicate evaluation, and hands it the
init order and the resolved execution graph before they run. A hook that
blocks pauses the run right there, which is how a step-by-step debugger UI
(e.g. the ``neofoam_ui`` DAG stepper) drives an unmodified solver.

No observer is installed by default; the hooks then cost one ``None`` check.

Example::

    class Printer(ExecutionObserver):
        def before_operation(self, op, ctx):
            print("next:", op.name)

    with observe(Printer()):
        run(argv)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from neofoam.framework.context import Context
    from neofoam.framework.initialization.init_step import InitStep
    from neofoam.framework.operations import Operation, Operations

__all__ = ["ExecutionObserver", "current", "observe", "set_observer"]


class ExecutionObserver:
    """Base class: every hook is a no-op, override the ones you need."""

    def init_steps(self, steps: list[InitStep]) -> None:
        """The init steps in execution order, before the first one runs."""

    def before_init_step(self, step: InitStep, context: dict[str, Any]) -> None:
        """``step`` is about to run; ``context`` holds the results so far."""

    def after_init_step(self, step: InitStep, context: dict[str, Any], value: Any) -> None:
        """``step`` produced ``value`` (not yet stored in ``context``)."""

    def operations(self, operations: Operations) -> None:
        """The resolved execution graph, before the time loop starts."""

    def before_operation(self, op: Operation, ctx: Context) -> None:
        """A sequential or conditional ``op`` is about to run."""

    def after_operation(self, op: Operation, ctx: Context) -> None:
        """A sequential or conditional ``op`` finished."""

    def before_loop_test(self, op: Operation, ctx: Context, iteration: int) -> None:
        """The predicate of loop ``op`` is about to decide on ``iteration`` (1-based)."""

    def after_loop_test(self, op: Operation, ctx: Context, iteration: int, result: bool) -> None:
        """The predicate of loop ``op`` returned ``result`` for ``iteration``."""


_observer: Optional[ExecutionObserver] = None


def current() -> Optional[ExecutionObserver]:
    """The installed observer, or ``None``."""
    return _observer


def set_observer(observer: Optional[ExecutionObserver]) -> Optional[ExecutionObserver]:
    """Install ``observer`` (``None`` removes it) and return the previous one."""
    global _observer
    previous, _observer = _observer, observer
    return previous


@contextmanager
def observe(observer: ExecutionObserver) -> Iterator[ExecutionObserver]:
    """Install ``observer`` for the duration of the ``with`` block."""
    previous = set_observer(observer)
    try:
        yield observer
    finally:
        set_observer(previous)
