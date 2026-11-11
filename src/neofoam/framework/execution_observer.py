# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Hooks through which a tool watches, or pauses, a solver run.

The installed :class:`ExecutionObserver` is called around every init step,
operation and loop test, and is handed the init order and the resolved graph
before they run. A hook that blocks pauses the run there: that is how a
step-by-step debugger UI drives an unmodified solver. Without an observer the
hooks cost one ``None`` check.

The observer is per thread (a :class:`~contextvars.ContextVar`); threads started
by an observed run do not inherit it, except the region threads of a coupled
run: there its hooks are called from several threads at once and must be thread safe.

Example::

    class Printer(ExecutionObserver):
        def before_operation(self, op, ctx):
            print("next:", op.name)

    with observe(Printer()):
        run(argv)
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from pydantic import BaseModel

    from neofoam.framework.context import Context
    from neofoam.framework.coupling.controller import CouplingController, CouplingOutcome
    from neofoam.framework.coupling.data import ExchangeSide
    from neofoam.framework.coupling.loop import CouplingOp
    from neofoam.framework.initialization.init_step import InitStep
    from neofoam.framework.operations import Operation, Operations

__all__ = ["ExecutionObserver", "current", "observe"]


class ExecutionObserver:
    """Base class: every hook is a no-op, override the ones you need."""

    def init_steps(self, steps: list[InitStep]) -> None:
        """The init steps in execution order, before the first one runs."""

    def before_init_step(self, step: InitStep, context: dict[str, Any]) -> None:
        """``step`` is about to run; ``context`` holds the results so far."""

    def after_init_step(self, step: InitStep, context: dict[str, Any], value: Any) -> None:
        """``step`` produced ``value`` (not yet stored in ``context``)."""

    def operations(self, operations: Operations, ctx: Context) -> None:
        """The resolved execution graph of ``ctx``, before the time loop starts; one per region."""

    def before_operation(self, op: Operation, ctx: Context) -> None:
        """A sequential or conditional ``op`` is about to run on ``ctx``."""

    def after_operation(self, op: Operation, ctx: Context) -> None:
        """A sequential or conditional ``op`` finished."""

    def before_loop_test(self, op: Operation, ctx: Context, iteration: int) -> None:
        """The predicate of loop ``op`` is about to decide on ``iteration`` (1-based)."""

    def after_loop_test(self, op: Operation, ctx: Context, iteration: int, result: bool) -> None:
        """The predicate of loop ``op`` returned ``result`` for ``iteration``."""

    def coupling_finished(self, coupling: CouplingController, outcome: CouplingOutcome) -> None:
        """A time step's coupling ended, as the leader's ``coupling`` decided; leader's thread."""

    def coupling_exchanged(self, coupling: CouplingOp, side: ExchangeSide, data: BaseModel) -> None:
        """The region's ``coupling`` op is about to hand ``side`` its mapped ``data``."""


_observer: ContextVar[Optional[ExecutionObserver]] = ContextVar(
    "neofoam_execution_observer", default=None
)


def current() -> Optional[ExecutionObserver]:
    """The observer installed in this thread, or ``None``."""
    return _observer.get()


@contextmanager
def observe(observer: ExecutionObserver) -> Iterator[ExecutionObserver]:
    """Install ``observer`` in this thread for the duration of the ``with`` block."""
    token = _observer.set(observer)
    try:
        yield observer
    finally:
        _observer.reset(token)
