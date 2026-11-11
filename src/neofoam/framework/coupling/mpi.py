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

from .comm_channel import DEFAULT_TIMEOUT, CouplingTimeout, _Mailbox, _pickled
from .data import ExchangeSide, _sent
from .loop import CouplingOp

if TYPE_CHECKING:
    from neofoam.framework.context import Context

    from .region import CouplingNode, RegionCoupling

# What travels between roots, pickled: (source region, kind, tag, message).
_MESSAGE, _CLOSED, _ABORTED = "message", "closed", "aborted"

# Seconds a waiting root sleeps between two polls: doubling from the first.
_FIRST_PAUSE, _LONGEST_PAUSE = 1e-5, 1e-3


class MpiCommChannel(_Mailbox):
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
        self._post(target, tag, _pickled(source, target, tag, (source, _MESSAGE, tag, message)))

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
        self, coupling: RegionCoupling, node: CouplingNode, sides: Sequence[ExchangeSide]
    ) -> IterativeOp:
        """The loop of ``node`` here: the coupling op on the root, a follower elsewhere."""
        if self.is_root:
            return _RootOp(coupling, node, sides, self)
        return _FollowerOp(node, sides, self)

    def time_step(self, agree: Callable[[Context], None]) -> Callable[[Context], None]:
        """``agree`` on the root; its time step on every process of the region."""

        def step(ctx: Context) -> None:
            if self.is_root:
                agree(ctx)
            ctx.time.delta_t = self.bcast(ctx.time.delta_t)

        return step


class _Patch:
    """A side's patch over the region's processes, its rows per process known once gathered."""

    def __init__(self, comm: RegionComm, side: ExchangeSide) -> None:
        self._comm = comm
        self._counts: Optional[tuple[int, ...]] = None
        self.points = self.gather(side.coordinates)

    def gather(self, rows: Any) -> Any:
        gathered = self._comm.gather(rows)
        if gathered is None:
            return None
        full, counts = gathered
        if self._counts is None:
            self._counts = counts
        return full

    def hand_over(self, side: ExchangeSide, whole: Optional[BaseModel]) -> None:
        """Hand ``side`` its rows of what the whole patch receives (``whole``, on the root)."""
        rows = {
            name: self._comm.scatter(None if whole is None else getattr(whole, name), self._counts)
            for name in side.receives.model_fields
        }
        side.receive(side.receives.model_validate(rows))


class _RootOp(CouplingOp):
    """The coupling op on a region's root: on the whole patches, its decision broadcast."""

    def __init__(
        self,
        coupling: RegionCoupling,
        node: CouplingNode,
        sides: Sequence[ExchangeSide],
        comm: RegionComm,
    ) -> None:
        self._comm = comm
        self._parts = [(side, _Patch(comm, side)) for side in sides]
        self._whole: dict[str, BaseModel] = {}
        super().__init__(coupling, node, [self._whole_side(*part) for part in self._parts])

    def _whole_side(self, side: ExchangeSide, patch: _Patch) -> ExchangeSide:
        def send() -> BaseModel:
            sent = side.send()
            fields = {name: patch.gather(getattr(sent, name)) for name in side.sends.model_fields}
            return side.sends.model_validate(fields)

        def receive(whole: BaseModel) -> None:
            self._whole[side.patch] = whole

        return replace(side, send=send, receive=receive, coordinates=patch.points)

    def _test(self, ctx: Context) -> bool:
        go_on = super()._test(ctx)
        self._comm.bcast((go_on, go_on and ctx.loops[-1].final))
        if go_on:
            for side, patch in self._parts:
                patch.hand_over(side, self._whole.pop(side.patch))
        return go_on


class _FollowerOp(IterativeOp):
    """A coupling op's place on a region's other processes: the root decides for them."""

    def __init__(self, node: CouplingNode, sides: Sequence[ExchangeSide], comm: RegionComm) -> None:
        super().__init__(self._test)
        self.node = node
        self._comm = comm
        self._parts = [(side, _Patch(comm, side)) for side in sides]

    def _test(self, ctx: Context) -> bool:
        iteration = ctx.loops[-1].iteration
        time = getattr(ctx.time, "value", None)
        for side, patch in self._parts:
            sent = _sent(side, self.node.name, iteration, time)
            for name in side.sends.model_fields:
                patch.gather(sent[name])
        go_on, final = self._comm.bcast(None)
        if go_on:
            ctx.mark_final(final)
            for side, patch in self._parts:
                patch.hand_over(side, None)
        return bool(go_on)
