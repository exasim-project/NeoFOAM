# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling on MPI processes: :class:`MpiCommChannel` between regions, :class:`RegionComm` within.

Only a region's root, rank 0 of its communicator, talks to other regions: its coupling
ops run on the whole patches, gathered from its processes and scattered back, while
the others run a follower op. Communicators are mpi4py's, duck-typed: only the
launcher imports mpi4py.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
from pydantic import BaseModel

from neofoam.framework.operations import IterativeOp

from ..coupled import CoupledPort
from ..data import Stream, sent
from ..iteration._loop_op import CouplingLoopOp
from ..location import Geometry
from .channel import DEFAULT_TIMEOUT, CouplingTimeout, Mailbox, pickled

if TYPE_CHECKING:
    from neofoam.framework.context import Context

    from ..exchanges import CouplingLoop, ParticipantExchanges

# What travels between roots, pickled: (source region, kind, tag, message).
_MESSAGE, _CLOSED, _ABORTED = "message", "closed", "aborted"

# Seconds a waiting root sleeps between two polls: doubling from the first.
_FIRST_PAUSE, _LONGEST_PAUSE = 1e-5, 1e-3


class MpiCommChannel(Mailbox):
    """A comm channel between the roots of regions on MPI processes: one MPI tag, a mailbox per tag.

    ``comm`` spans every process of the run (a ``Dup`` of MPI_COMM_WORLD); ``roots``
    maps each region of the group to its root's rank in it. A deadlock between processes
    shows as a timeout only. Not thread safe.
    """

    TAG = 7

    def __init__(
        self,
        comm: Any,
        roots: Mapping[str, int],
        region: str,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(timeout)
        if region not in roots:
            raise ValueError(f"region {region} has no root among {dict(roots)}")
        if roots[region] != comm.rank:
            raise ValueError(
                f"region {region}'s root is rank {roots[region]}, not this process "
                f"(rank {comm.rank}); only a region's root talks to the other regions"
            )
        self.comm = comm
        self.roots = dict(roots)
        self.region = region
        self._sends: list[tuple[str, str, Any]] = []
        self._left = False
        self._pause = _FIRST_PAUSE

    def start(self, regions: Iterable[str]) -> None:
        pass

    def send(self, source: str, target: str, tag: str, message: Any) -> None:
        self._own(source, "sends as")
        if target not in self.roots:
            raise ValueError(f"region {source} sends '{tag}' to {target}, which is no region")
        self._post(target, tag, pickled(source, target, tag, (source, _MESSAGE, tag, message)))

    def receive(self, source: str, target: str, tag: str) -> Any:
        self._own(target, "receives as")
        self._drain()
        return self._take(source, target, tag)

    def close(self, region: str) -> None:
        """This root's region leaves; wait until every other root closed and every send completed.

        Raises :class:`CouplingTimeout` after ``timeout`` s; closing twice is a no-op.
        """
        self._own(region, "closes")
        if self._left:
            return
        self._left = True
        self._tell(_CLOSED)
        since = time.monotonic()
        while self._missing() or self._pending():
            left = None if self.timeout is None else since + self.timeout - time.monotonic()
            if left is not None and left <= 0:
                raise CouplingTimeout(self._close_report())
            self._wait(left)

    def abort(self, regions: Iterable[str]) -> None:
        """``regions`` leave; this root tells the others when its own region is one of them."""
        regions = set(regions)
        self._aborted |= regions - {self.region}
        if self.region in regions and not self._left:
            self._left = True
            self._tell(_ABORTED)

    def _wait(self, left: Optional[float]) -> None:
        if self._drain():
            self._pause = _FIRST_PAUSE
            return
        time.sleep(self._pause if left is None else min(self._pause, left))
        self._pause = min(2 * self._pause, _LONGEST_PAUSE)

    def _own(self, region: str, what: str) -> None:
        if region != self.region:
            raise ValueError(
                f"the MPI comm channel of region {self.region} {what} region {region}; each "
                "region's root has its own"
            )

    def _tell(self, kind: str) -> None:
        """Tell every other root that this region left."""
        payload = pickle.dumps((self.region, kind, None, None))
        for other in self.roots:
            if other != self.region:
                self._post(other, kind, payload)

    def _post(self, target: str, tag: str, payload: bytes) -> None:
        request = self.comm.isend(payload, dest=self.roots[target], tag=self.TAG)
        self._sends.append((target, tag, request))
        self._pending()

    def _pending(self) -> list[tuple[str, str, Any]]:
        """The sends not complete yet, tested without blocking."""
        self._sends = [send for send in self._sends if not send[2].Test()]
        return self._sends

    def _missing(self) -> list[str]:
        """The other roots that have not left yet."""
        left = self._closed | self._aborted | {self.region}
        return sorted(set(self.roots) - left)

    def _drain(self) -> bool:
        """Take every message that arrived into the mailboxes; whether one arrived."""
        arrived = False
        while self.comm.iprobe(tag=self.TAG):
            source, kind, tag, message = pickle.loads(self.comm.recv(tag=self.TAG))
            if kind == _MESSAGE:
                self._messages[(source, self.region, tag)].append(message)
            elif kind == _CLOSED:
                self._closed.add(source)
            else:
                self._aborted.add(source)
            arrived = True
        return arrived

    def _close_report(self) -> str:
        sends = ", ".join(f"{self.region}→{target} '{tag}'" for target, tag, _ in self._sends)
        return self._report(
            f"a coupling timeout: region {self.region} waited {self.timeout:g} s to close "
            f"(commChannelTimeout); not closed: {', '.join(self._missing()) or 'none'}; "
            f"sends not complete: {sends or 'none'}"
        )


class RegionComm:
    """The processes of a region on several, for its coupling: the root speaks for them.

    ``comm`` is the region's mpi4py communicator; rank 0 is the root. Every
    method is collective: each process of the region calls it, in the same order.
    """

    def __init__(self, comm: Any) -> None:
        self.comm = comm

    @property
    def is_root(self) -> bool:
        """Whether this process is the region's root, rank 0."""
        return int(self.comm.rank) == 0

    def gather(self, local: Any) -> Optional[tuple[Any, Optional[tuple[int, ...]]]]:
        """Every process's rows of ``local`` in rank order, with the rows of each, on the root.

        ``None`` on the other processes; ``(None, None)`` when every process
        passes ``None``. Raises ``ValueError`` when only some do.
        """
        parts = self.comm.gather(None if local is None else np.asarray(local, dtype=float), root=0)
        if not self.is_root:
            return None
        if all(part is None for part in parts):
            return None, None
        if any(part is None for part in parts):
            ranks = [rank for rank, part in enumerate(parts) if part is None]
            raise ValueError(f"the processes {ranks} of the region have no data, the others have")
        counts = tuple(len(part) for part in parts)
        filled = [part for part in parts if len(part)]
        return (np.concatenate(filled) if filled else parts[0]), counts

    def scatter(self, full: Any, counts: Optional[Sequence[int]]) -> Any:
        """Each process's ``counts[rank]`` rows of the root's ``full``, in order."""
        parts = None
        if self.is_root:
            full = np.asarray(full, dtype=float)
            if counts is None or len(counts) != self.comm.size or sum(counts) != len(full):
                raise ValueError(
                    f"cannot scatter {len(full)} rows over the region's {self.comm.size} "
                    f"processes by the counts {counts}"
                )
            parts = np.split(full, np.cumsum(counts)[:-1])
        return self.comm.scatter(parts, root=0)

    def bcast(self, value: Any) -> Any:
        """The root's ``value``, on every process."""
        return self.comm.bcast(value, root=0)

    def coupling_op(
        self,
        coupling: ParticipantExchanges,
        loop: CouplingLoop,
        ports: Sequence[CoupledPort],
        geometries: Mapping[str, Geometry],
    ) -> IterativeOp:
        """The loop of ``loop`` here: the coupling op on the root, a follower elsewhere."""
        parts = [_Port(self, port, geometries[port.name]) for port in ports]
        if self.is_root:
            return _RootOp(coupling, loop, parts, self)
        return _FollowerOp(loop, parts, self)

    def time_step(self, agree: Callable[[Context], None]) -> Callable[[Context], None]:
        """``agree`` on the root; its time step on every process of the region."""

        def step(ctx: Context) -> None:
            if self.is_root:
                agree(ctx)
            ctx.time.delta_t = self.bcast(ctx.time.delta_t)

        return step


def _spread(port: CoupledPort) -> bool:
    """Whether the port's rows are spread over the processes: a patch or the cells, no Global."""
    return not ({"value", "probe"} & set(port.location.entities))


class _Port:
    """A port over the region's processes: on the root its whole geometry, rows per process.

    A patch's or the cells' rows are gathered to the root and scattered back; a
    Global port sends the root's values and every process receives them.
    """

    def __init__(self, comm: RegionComm, port: CoupledPort, geometry: Geometry) -> None:
        self._comm = comm
        self.port = port
        self._counts: Optional[tuple[int, ...]] = None
        self.geometry = geometry
        if _spread(port):
            points, measures = self.gather(geometry.points), self.gather(geometry.measures)
            self.geometry = Geometry.of(points, measures) if comm.is_root else geometry

    def gather(self, rows: Any) -> Any:
        """Every process's ``rows`` in rank order on the root (``None`` elsewhere)."""
        gathered = self._comm.gather(rows)
        if gathered is None:
            return None
        full, counts = gathered
        if self._counts is None and counts is not None:
            self._counts = counts
        return full

    def gather_sent(self, stream: Stream, sent: BaseModel) -> BaseModel:
        """What the whole port sends of ``stream``: the processes' rows, or the root's value."""
        if not _spread(self.port):
            return sent
        rows = {name: self.gather(getattr(sent, name)) for name in stream.names}
        whole: BaseModel = stream.model.model_validate(rows)
        return whole

    def hand_over(self, whole: Mapping[str, BaseModel]) -> None:
        """Hand each receiving stream its rows of what the whole port receives (on the root)."""
        for stream in self.port.receives:
            received = whole.get(stream.key)
            if _spread(self.port):
                rows = {
                    name: self._comm.scatter(
                        None if received is None else getattr(received, name), self._counts
                    )
                    for name in stream.names
                }
                stream.method(stream.model.model_validate(rows))
            else:
                stream.method(self._comm.bcast(received))


class _RootOp(CouplingLoopOp):
    """The coupling op on a region's root: on the whole ports, its decision broadcast."""

    def __init__(
        self,
        coupling: ParticipantExchanges,
        loop: CouplingLoop,
        parts: Sequence[_Port],
        comm: RegionComm,
    ) -> None:
        self._comm = comm
        self._parts = list(parts)
        self._whole: dict[str, dict[str, BaseModel]] = {part.port.name: {} for part in parts}
        super().__init__(
            coupling,
            loop,
            [self._whole_port(part) for part in parts],
            {part.port.name: part.geometry for part in parts},
        )

    def _whole_port(self, part: _Port) -> CoupledPort:
        whole = self._whole[part.port.name]

        def sends(stream: Stream) -> Stream:
            return Stream(stream.model, lambda: part.gather_sent(stream, stream.method()))

        def receives(stream: Stream) -> Stream:
            return Stream(stream.model, lambda data: whole.__setitem__(stream.key, data))

        return replace(
            part.port,
            sends=tuple(sends(stream) for stream in part.port.sends),
            receives=tuple(receives(stream) for stream in part.port.receives),
        )

    def _test(self, ctx: Context) -> bool:
        go_on = super()._test(ctx)
        self._comm.bcast((go_on, go_on and ctx.loops[-1].final))
        if go_on:
            for part in self._parts:
                part.hand_over(self._whole[part.port.name])
                self._whole[part.port.name].clear()
        return go_on


class _FollowerOp(IterativeOp):
    """A coupling op's place on a region's other processes: the root decides for them."""

    def __init__(self, loop: CouplingLoop, parts: Sequence[_Port], comm: RegionComm) -> None:
        super().__init__(self._test)
        self.loop = loop
        self._comm = comm
        self._parts = list(parts)

    def _test(self, ctx: Context) -> bool:
        when = (ctx.loops[-1].iteration, getattr(ctx.time, "value", None))
        for part in self._parts:
            for stream in part.port.sends:
                data = sent(stream, part.port, self.loop.name, when)
                if _spread(part.port):
                    for name in stream.names:
                        part.gather(data[name])
        go_on, final = self._comm.bcast(None)
        if go_on:
            ctx.mark_final(final)
            for part in self._parts:
                part.hand_over({})
        return bool(go_on)
