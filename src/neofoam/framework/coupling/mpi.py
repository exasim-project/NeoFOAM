# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling across MPI processes: :class:`MpiTransport` between regions, :class:`RegionComm` within.

Only a region's root, rank 0 of its communicator, talks to other regions. Its
coupling ops run on the whole patches: what the sides of every process send is
gathered to the root, what they receive is scattered back. The other processes
run a follower op in place of each coupling op. Communicators are mpi4py's,
duck-typed (the launcher imports mpi4py).

Example::

    channel = MPI.COMM_WORLD.Dup()                       # every process
    comm = MPI.COMM_WORLD.Split(color, key=world.rank)    # the region's processes
    comm_channel = MpiTransport(channel, {"left": 0, "right": 2}, region) if comm.rank == 0 else None
    coupling = replace(coupling, comm_channel=comm_channel, comm=RegionComm(comm))
"""

from __future__ import annotations

import pickle
import time
from collections import defaultdict, deque
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
from pydantic import BaseModel

from neofoam.framework.operations import IterativeOp

from .comm_channel import DEFAULT_TIMEOUT, CommChannelClosed, CouplingDeadlock, CouplingTimeout
from .data import ExchangeSide, _sent
from .loop import CouplingOp

if TYPE_CHECKING:
    from neofoam.framework.context import Context

    from .region import CouplingNode, RegionCoupling

# What travels between roots, pickled: (source region, kind, tag, message).
_MESSAGE, _CLOSED, _ABORTED = "message", "closed", "aborted"

# Seconds a waiting root sleeps between two polls: doubling from the first.
_FIRST_PAUSE, _LONGEST_PAUSE = 1e-5, 1e-3


class MpiTransport:
    """A comm channel between the roots of regions on MPI processes: one MPI tag, a mailbox per tag.

    ``channel`` is a communicator of every process of the run (a ``Dup`` of
    MPI_COMM_WORLD, so no other message meets the coupling's), ``roots`` the
    rank in it of each region's root, and ``region`` the region of this
    process, which must be its root. Every message goes as one pickle under the
    one MPI tag :attr:`TAG`, the string tag inside it; a receive takes every
    message that arrived into a mailbox per ``(source, tag)`` and returns the
    first of its own. MPI does not overtake messages between two processes on
    one communicator and tag, so the messages of one ``(source, target)``
    arrive in the order they were sent, of every tag (more than the contract
    of :class:`~neofoam.framework.coupling.CommChannel`). A send never blocks
    (``isend``); a message that does not pickle raises ``TypeError`` at once.

    A receive raises :class:`CouplingDeadlock` once ``source`` finished
    (:meth:`close`) and :class:`CommChannelClosed` once it failed (:meth:`abort`)
    with no such message waiting, and :class:`CouplingTimeout` when it waits
    longer than ``timeout`` seconds (``None`` or ``0``: without end), naming
    what it waits for and the messages that arrived but were not received.
    A deadlock between processes is told by that timeout only. Raises
    ``ValueError`` for a negative ``timeout``, and on a send or receive of
    another region than ``region``. Not thread safe.
    """

    #: The one MPI tag of the coupling's messages.
    TAG = 7

    def __init__(
        self,
        channel: Any,
        roots: Mapping[str, int],
        region: str,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
    ) -> None:
        if timeout is not None and timeout < 0:
            raise ValueError(f"a comm channel timeout is at least 0 seconds, not {timeout}")
        if region not in roots:
            raise ValueError(f"region {region} has no root among {dict(roots)}")
        if roots[region] != channel.rank:
            raise ValueError(
                f"region {region}'s root is rank {roots[region]}, not this process "
                f"(rank {channel.rank}); only a region's root talks to the other regions"
            )
        self.channel = channel
        self.roots = dict(roots)
        self.region = region
        self.timeout = timeout or None
        self._mail: dict[tuple[str, str], deque[Any]] = defaultdict(deque)
        self._closed: set[str] = set()
        self._aborted: set[str] = set()
        self._requests: list[Any] = []
        self._left = False

    def start(self, regions: Iterable[str]) -> None:
        pass

    def send(self, source: str, target: str, tag: str, message: Any) -> None:
        self._own(source, "sends as")
        if target not in self.roots:
            raise ValueError(f"region {source} sends '{tag}' to {target}, which is no region")
        try:
            payload = pickle.dumps((source, _MESSAGE, tag, message), protocol=5)
        except Exception as error:
            raise TypeError(
                f"the message '{tag}' from region {source} to region {target} is not plain "
                f"data, it does not pickle: {error}"
            ) from error
        self._post(target, payload)

    def receive(self, source: str, target: str, tag: str) -> Any:
        self._own(target, "receives as")
        key = (source, tag)
        since = time.monotonic()
        pause = _FIRST_PAUSE
        while True:
            if self._drain():
                pause = _FIRST_PAUSE
            if self._mail[key]:
                return self._mail[key].popleft()
            if source in self._aborted:
                raise CommChannelClosed(
                    f"region {target} waits for '{tag}' from region {source}, "
                    "which left the coupling"
                )
            if source in self._closed:
                raise CouplingDeadlock(
                    self._report(
                        f"a coupling deadlock: region {target} waits for '{tag}' from region "
                        f"{source}, which finished",
                        source,
                        tag,
                        since,
                    )
                )
            if self.timeout is not None and time.monotonic() - since > self.timeout:
                raise CouplingTimeout(
                    self._report(
                        f"a coupling timeout: region {target} waited {self.timeout:g} s for "
                        f"'{tag}' from region {source} (commChannelTimeout)",
                        source,
                        tag,
                        since,
                    )
                )
            time.sleep(pause)
            pause = min(2 * pause, _LONGEST_PAUSE)

    def close(self, region: str) -> None:
        """Region ``region`` (this root's) leaves; waits until every other root left too.

        Tells every other root first, so a partner waiting for this region fails
        at once (:class:`CouplingDeadlock`). Then it takes the messages still
        coming until each other root closed (or aborted): every message
        between the roots is received, every send complete, before MPI is
        finalized. Closing again does nothing.
        """
        self._own(region, "closes")
        if self._left:
            return
        self._left = True
        self._tell(_CLOSED)
        pause = _FIRST_PAUSE
        others = set(self.roots) - {self.region}
        while not others <= self._closed | self._aborted:
            if not self._drain():
                time.sleep(pause)
                pause = min(2 * pause, _LONGEST_PAUSE)
        for request in self._requests:
            request.Wait()
        self._requests.clear()

    def abort(self, regions: Iterable[str]) -> None:
        """Every region of ``regions`` leaves; this root tells the others if its own is one.

        A root speaks for its region only: the others learn of a failed region
        from it (its receives raise :class:`CommChannelClosed`). Under multiRun the
        process aborts every process right after (``MPI_Abort``).
        """
        regions = set(regions)
        self._aborted |= regions - {self.region}
        if self.region in regions and not self._left:
            self._left = True
            self._tell(_ABORTED)

    def _own(self, region: str, what: str) -> None:
        if region != self.region:
            raise ValueError(
                f"the MPI comm channel of region {self.region} {what} region {region}; each "
                "region's root has its own"
            )

    def _tell(self, kind: str) -> None:
        """Tell every other root that this region left (``kind``: closed or aborted)."""
        payload = pickle.dumps((self.region, kind, None, None))
        for other in self.roots:
            if other != self.region:
                self._post(other, payload)

    def _post(self, target: str, payload: bytes) -> None:
        self._requests.append(self.channel.isend(payload, dest=self.roots[target], tag=self.TAG))
        self._requests = [request for request in self._requests if not request.Test()]

    def _drain(self) -> bool:
        """Take every message that arrived into the mailboxes; whether one arrived."""
        arrived = False
        while self.channel.iprobe(tag=self.TAG):
            source, kind, tag, message = pickle.loads(self.channel.recv(tag=self.TAG))
            if kind == _MESSAGE:
                self._mail[(source, tag)].append(message)
            elif kind == _CLOSED:
                self._closed.add(source)
            else:
                self._aborted.add(source)
            arrived = True
        return arrived

    def _report(self, headline: str, source: str, tag: str, since: float) -> str:
        """``headline``, what this root waits for since when, the messages it did not receive."""
        lines = [
            headline + ":",
            f"  {self.region} waits for '{tag}' from {source} "
            f"(for {time.monotonic() - since:.1f} s)",
        ]
        left = sorted(self._closed | self._aborted)
        if left:
            lines.append(f"  left the coupling: {', '.join(left)}")
        undelivered = [
            f"{origin}→{self.region} '{name}'" + (f" ({len(queue)})" if len(queue) > 1 else "")
            for (origin, name), queue in sorted(self._mail.items())
            if queue
        ]
        lines.append(f"undelivered messages: {', '.join(undelivered) or 'none'}")
        return "\n".join(lines)


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
