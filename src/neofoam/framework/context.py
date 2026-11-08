# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Simulation context: shared field/model state passed through operations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, NamedTuple, Optional

from pydantic import BaseModel, Field


class FieldUpdates(dict[str, Any]):
    """
    A dictionary that collects the pending field updates for Context.fields.
    """

    pass


_initializing: ContextVar[Optional[str]] = ContextVar("neofoam_initializing_region", default=None)


@contextmanager
def initializing_region(region: Optional[str]) -> Iterator[None]:
    """Initialize the region ``region`` in the ``with`` block.

    A region's ``SolverRuntime.initialize`` enters it: the init steps it runs
    carry the region (``InitStep.region``), and a :class:`Context` built in it
    gets ``region``. ``None`` keeps the enclosing region.
    """
    if region is None:
        yield
        return
    token = _initializing.set(region)
    try:
        yield
    finally:
        _initializing.reset(token)


def initializing() -> Optional[str]:
    """The region being initialized in this thread (see :func:`initializing_region`)."""
    return _initializing.get()


class LoopEntry(NamedTuple):
    """A running loop, an entry of :attr:`Context.loops`.

    ``iteration`` counts passes from 0; ``final`` is set by the loop's test
    (:meth:`Context.mark_final`).
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
    # the loops running now, outermost first
    loops: list[LoopEntry] = Field(default_factory=list)
    # ``final`` operations to run in every pass too, e.g. to watch them while debugging
    run_every_pass: set[str] = Field(default_factory=set)
    # the region of a multi-region case this context belongs to; None for one region
    region: Optional[str] = Field(default_factory=initializing)

    def mark_final(self, final: bool = True) -> None:
        """Mark the pass the innermost loop is about to run as its last; call from its test.

        The next pass is not final again until the test marks it.
        """
        self.loops[-1] = self.loops[-1]._replace(final=final)
