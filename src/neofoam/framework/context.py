# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Simulation context: shared field/model state passed through operations."""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

from pydantic import BaseModel, Field


class FieldUpdates(dict[str, Any]):
    """
    A dictionary that collects the pending field updates for Context.fields.
    """

    pass


class LoopEntry(NamedTuple):
    """A running loop, an entry of :attr:`Context.loops`.

    Read it from an operation or loop test that depends on the current pass; for
    a run-once or last-pass operation prefer the ``first``/``final`` marks.
    Immutable: ``iteration`` counts passes from 0 and the running loop replaces its
    entry each pass; ``final`` is set by the loop's test (:meth:`Context.mark_final`).

    Example::

        if ctx.loops[-1].first:
            store_old_iterate(ctx)
    """

    name: Optional[str]
    iteration: int
    wrapping: bool
    final: bool = False

    @property
    def first(self) -> bool:
        """Whether the loop is in its first pass."""
        return self.iteration == 0


class Context(BaseModel):
    """
    The Context object holds the state of the framework at a given point in time.
    It contains fields and models that are used by various components of the framework.

    The relevant fields or models are injected into the operations

    ``loops`` holds the running loops, outermost first. ``run_every_pass`` names
    ``final`` operations that also run in every other pass, e.g. to watch them
    while debugging.
    """

    model_config = {"arbitrary_types_allowed": True}
    fields: dict[str, Any]
    models: dict[str, Any]
    mesh: Any = None
    # the pure-Python LoopState (time, deltaT, index, write flag); advanced by
    # the solutionLoop engine. The backend Foam::Time (if any) is hidden.
    time: Any = None
    # names of fields (keys of ``fields``) flagged for persistence via
    # ``field(..., write=True)``; a per-field write backend writes exactly these.
    write_fields: set[str] = Field(default_factory=set)
    loops: list[LoopEntry] = Field(default_factory=list)
    run_every_pass: set[str] = Field(default_factory=set)
    # set by SolverRuntime.initialize; None for a single-region case
    region: Optional[str] = None

    def mark_final(self, final: bool = True) -> None:
        """Mark the pass the innermost loop is about to run as its last.

        Call it from that loop's test, when the test knows the coming pass ends the
        loop; a loop that never calls it never runs its ``final`` operations. The
        mark lasts one pass: the next is not final until the test marks it again.

        Example::

            def converged_test(ctx: Context) -> bool:
                ctx.mark_final(residual(ctx) < tolerance)
                return ctx.loops[-1].iteration < max_passes
        """
        self.loops[-1] = self.loops[-1]._replace(final=final)
