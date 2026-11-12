# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — operations take ``ctx: Context``.

"""In-situ entries of a multiRun: ``insitu:`` in the case's ``system/postProcess.yaml``.

The entries are the case's, not a region's: every region steps an ``insitu``
operation last in its time loop (after its post-processing), and on an entry's
write steps the regions meet there, so the entry draws them all at one time
(:meth:`~neofoam.postprocess.insitu.InSitu.execute_regions`). On threads the
regions wait for each other (a barrier; one of them draws); under MPI each
process hands in its own region and the drawing is collective. The regions must
step the same times. A region's own ``insitu`` entries are refused in a
multiRun, its tables are not::

    # system/postProcess.yaml (case root)
    insitu:
      - {name: view, type: catalyst, field: T}
"""

import threading
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from neofoam.framework.context import Context
from neofoam.framework.coupling import CommChannelClosed
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata
from neofoam.postprocess.config import load_config, spec_file
from neofoam.postprocess.insitu import InSitu, InSituEntry
from neofoam.postprocess.table import tables_for_case
from neofoam.solver.regions import region_case_dir

#: The region's loop whose last operation the ``insitu`` op follows.
_TIME_LOOP = "time_loop"

Render = Callable[[dict[str, Optional[Context]]], None]


def declared_insitu(regions: Sequence[str]) -> list[InSituEntry]:
    """The case's in-situ entries; ``ValueError`` for a region's own or a case-level table.

    Example::

        entries = declared_insitu(["fluid", "solid"])
    """
    for region in regions:
        case_dir = region_case_dir(region)
        if spec_file(case_dir) is not None and load_config(case_dir).insitu:
            raise ValueError(
                f"multiRun: region {region} declares insitu entries in its "
                "system/postProcess.yaml; in a multiRun they belong in the case's "
                "system/postProcess.yaml, where they draw every region together"
            )
    case = tables_for_case(Path("."))
    if case.tables:
        raise ValueError(
            "multiRun: the case's system/postProcess holds insitu entries only; "
            "a table belongs in its region's system/postProcess.yaml"
        )
    return case.insitu


def _same_time(times: dict[str, float]) -> None:
    if len(set(times.values())) > 1:
        raise ValueError(
            f"multiRun insitu: the regions are at different times {times}; "
            "they are drawn together on the same time steps only"
        )


class ThreadMeeting:
    """The regions of one process, each on its thread: they meet at a barrier, one draws."""

    def __init__(self, regions: Sequence[str], timeout: Optional[float]) -> None:
        self._barrier = threading.Barrier(len(regions), timeout=timeout)
        self._contexts: dict[str, Optional[Context]] = dict.fromkeys(regions)
        self._running = len(regions)
        self._lock = threading.Lock()

    def meet(self, region: str, ctx: Context, render: Render) -> None:
        """Wait until every region is here; then one of them calls ``render``."""
        self._contexts[region] = ctx
        try:
            if self._barrier.wait() == 0:
                try:
                    _same_time({r: float(c.time.value) for r, c in self._contexts.items() if c})
                    render(dict(self._contexts))
                except BaseException:
                    self._barrier.abort()
                    raise
            self._barrier.wait()
        except threading.BrokenBarrierError:
            # a consequence: another region failed (or timed out) on its way here
            raise CommChannelClosed(
                f"region {region}: the other regions did not meet at the in-situ step "
                f"of time {ctx.time.value}"
            ) from None

    def abort(self) -> None:
        """A region failed: no region waits for it."""
        self._barrier.abort()

    def last_out(self) -> bool:
        """Count a finished region; True for the last one."""
        with self._lock:
            self._running -= 1
            return self._running == 0


class MpiMeeting:
    """One region per process group: every process hands in its own region, collectively."""

    def __init__(self, world: Any, regions: Sequence[str]) -> None:
        self._world = world
        self._regions = list(regions)

    def meet(self, region: str, ctx: Context, render: Render) -> None:
        _same_time(dict(self._world.allgather((region, float(ctx.time.value)))))
        render({r: ctx if r == region else None for r in self._regions})

    def abort(self) -> None:
        pass  # an error aborts every process (MPI_Abort)

    def last_out(self) -> bool:
        return True  # each process finalizes its own session


class MultiRunInSitu:
    """The case's in-situ entries, shared by the regions of this process.

    Example::

        insitu = MultiRunInSitu(entries, ThreadMeeting(regions, timeout))
        ctx.models["multirun_insitu"] = insitu.region("fluid")
    """

    def __init__(self, entries: list[InSituEntry], meeting: Any) -> None:
        self.entries = [e.plugin.model_copy(deep=True) for e in entries]
        self._write_controls = [e.write_control for e in entries]
        self.meeting = meeting

    def region(self, region: str) -> "RegionInSitu":
        """``region``'s part: its ``insitu`` operation, on its own copy of the cadences."""
        return RegionInSitu(self, region, [wc.model_copy(deep=True) for wc in self._write_controls])

    def finalize(self) -> None:
        if self.meeting.last_out():
            for plugin in self.entries:
                plugin.finalize()


class RegionInSitu:
    """A region's model: adds the ``insitu`` op to its graph and finalizes with the run.

    A :class:`~neofoam.framework.solver.runtime.GraphContribution` and a
    :class:`~neofoam.framework.solver.runtime.Finalizer` of ``ctx.models``.
    """

    def __init__(self, shared: MultiRunInSitu, region: str, write_controls: list[Any]) -> None:
        self.shared = shared
        self.region = region
        self._write_controls = write_controls

    def step(self, ctx: Context) -> None:
        """Meet the other regions on the entries' write steps and draw them together."""
        due: list[InSitu] = [
            plugin
            for plugin, write_control in zip(self.shared.entries, self._write_controls)
            if write_control.should_write(ctx.time)
        ]
        if not due:
            return

        def render(regions: dict[str, Optional[Context]]) -> None:
            for plugin in due:
                plugin.execute_regions(regions)

        self.shared.meeting.meet(self.region, ctx, render)

    def graph_operations(self, ctx: Context, graph: Sequence[Operation]) -> Operations:
        loop = next((op for op in graph if op.operation_name == _TIME_LOOP), None)
        after = loop.sub_operations[-1].operation_name if loop and loop.sub_operations else None
        if after is None:
            raise ValueError(
                f"multiRun insitu: region {self.region}'s graph has no {_TIME_LOOP} to draw in"
            )
        return Operations(
            [
                Operation(
                    func=SequentialOp(self.step),
                    metadata=OperationMetadata(op_name="insitu", depends_on=[after]),
                )
            ]
        )

    def check_graph(self, graph: Operations) -> None:
        pass

    def finalize(self) -> None:
        self.shared.finalize()
