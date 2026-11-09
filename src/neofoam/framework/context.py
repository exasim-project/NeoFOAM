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

    ``iteration`` counts the loop's passes from 0 (the time loop: its time
    steps); ``first`` is whether it is in its first pass. ``wrapping`` is whether
    the loop is a wrapping loop (``OperationMetadata.wraps``, e.g. a coupling
    node), placed around operations of the time step. ``final`` is whether the
    pass is the loop's last one: its loop test says so (:meth:`Context.mark_final`,
    e.g. a coupling node's leader); ``False`` for a loop whose test does not.
    """

    name: Optional[str]
    iteration: int
    first: bool
    wrapping: bool
    final: bool = False


class Context(BaseModel):
    """
    The Context object holds the state of the framework at a given point in time.
    It contains fields and models that are used by various components of the framework.

    The relevant fields or models are injected into the operations
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
    # the loops running now, outermost first: each loop op adds its entry while it
    # runs and updates it before each pass (``Operation``)
    loops: list[LoopEntry] = Field(default_factory=list)
    # names (``Operation.name``) of operations marked ``final`` to run in every pass
    # of their wrapping loops instead, e.g. to watch them while debugging
    run_every_pass: set[str] = Field(default_factory=set)

    def mark_final(self, final: bool = True) -> None:
        """Mark the pass the innermost running loop is about to run as its last one.

        Called by a loop's test before the pass (it sets ``final`` of the loop's
        entry on :attr:`loops`); the entry of the next pass is not final again
        until the test marks it.
        """
        self.loops[-1] = self.loops[-1]._replace(final=final)
