# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling across MPI processes: :class:`MpiTransport` between regions, :class:`RegionComm` within.

Under ``mpirun`` every region runs on processes of its own (multiRun splits
MPI_COMM_WORLD by region). Only a region's **root**, rank 0 of its
communicator, talks to the other regions: the roots exchange the coupling's
messages over an :class:`MpiTransport`, with today's protocol. The other
processes of a region (its **followers**) hand their part of each side to the
root and get theirs back, over the region's :class:`RegionComm`: the root
gathers what every process sends, runs the exchange, scatters what each process
receives and broadcasts the leader's decision and the agreed time step. A
region on one process has a :class:`RegionComm` of size 1, which does nothing:
the coupling of threads.

Both take an mpi4py communicator (duck-typed: mpi4py is imported by the
launcher only, the extra ``neofoam[mpi]``).

Example::

    channel = MPI.COMM_WORLD.Dup()                       # every process
    comm = MPI.COMM_WORLD.Split(color, key=world.rank)    # the region's processes
    comm_channel = (MpiTransport(channel, {"left": 0, "right": 2}, region)
                 if comm.rank == 0 else FollowerTransport(region))
    region_comm = RegionComm(comm)
"""

from __future__ import annotations

import pickle
import time
from collections import defaultdict, deque
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .comm_channel import (
    DEFAULT_TIMEOUT,
    CommChannel,
    CommChannelClosed,
    CouplingDeadlock,
    CouplingTimeout,
)

# What travels between roots, pickled: (source region, kind, tag, message).
_MESSAGE, _CLOSED, _ABORTED = "message", "closed", "aborted"

# How long a waiting receive sleeps between two polls, in seconds: doubling from the first.
_FIRST_PAUSE, _LONGEST_PAUSE = 1e-5, 1e-3


class MpiTransport(CommChannel):
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


class FollowerTransport(CommChannel):
    """The comm channel of a region's follower (not its root): it talks to no other region.

    Its coupling goes over the region's :class:`RegionComm` to the root; a send
    or receive raises ``RuntimeError``, :meth:`close` does nothing.
    """

    def __init__(self, region: str) -> None:
        self.region = region

    def send(self, source: str, target: str, tag: str, message: Any) -> None:
        raise RuntimeError(self._refusal(f"send '{tag}' to region {target}"))

    def receive(self, source: str, target: str, tag: str) -> Any:
        raise RuntimeError(self._refusal(f"receive '{tag}' from region {source}"))

    def close(self, region: str) -> None:
        pass

    def _refusal(self, what: str) -> str:
        return (
            f"a process of region {self.region} that is not its root cannot {what}; only the "
            "region's root talks to other regions"
        )


class RegionComm:
    """The processes of one region, for its coupling: the root gathers, scatters and broadcasts.

    ``comm`` is the region's mpi4py communicator, ``None`` for a region on one
    process: then every method hands back what it gets (size 1, the coupling of
    threads). Rank 0 is the root. Every method is collective: each process of
    the region calls it, in the same order.
    """

    def __init__(self, comm: Any = None) -> None:
        self.comm = comm

    @property
    def size(self) -> int:
        """The number of processes of the region."""
        return 1 if self.comm is None else int(self.comm.size)

    @property
    def rank(self) -> int:
        """This process's rank in the region."""
        return 0 if self.comm is None else int(self.comm.rank)

    @property
    def is_root(self) -> bool:
        """Whether this process is the region's root, rank 0."""
        return self.rank == 0

    def gather(self, local: Any) -> Optional[tuple[Any, Optional[tuple[int, ...]]]]:
        """Every process's rows of ``local`` (axis 0), in rank order, on the root.

        Returns ``(full, counts)`` on the root (``counts``: the rows of each
        process, 0 allowed), ``None`` elsewhere. ``local`` ``None`` on every
        process gives ``(None, None)``; raises ``ValueError`` when some
        processes pass ``None`` and others do not.
        """
        if self.size == 1:
            return local, None if local is None else (len(local),)
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
        """Each process's rows of the root's ``full`` (axis 0): ``counts[rank]`` of them, in order.

        ``full`` and ``counts`` matter on the root only. Raises ``ValueError`` on
        the root when ``counts`` do not add up to the rows of ``full``.
        """
        if self.size == 1:
            return full
        parts = None
        if self.is_root:
            full = np.asarray(full, dtype=float)
            if counts is None or len(counts) != self.size or sum(counts) != len(full):
                raise ValueError(
                    f"cannot scatter {len(full)} rows over the region's {self.size} processes "
                    f"by the counts {counts}"
                )
            parts = np.split(full, np.cumsum(counts)[:-1])
        return self.comm.scatter(parts, root=0)

    def bcast(self, value: Any) -> Any:
        """The root's ``value``, on every process."""
        if self.size == 1:
            return value
        return self.comm.bcast(value, root=0)

    def agree(self, what: str, value: Any) -> Any:
        """``value`` of rank 0, checked equal on every process (on the root).

        For a value that is one for the whole region (a plain field of a side).
        Raises ``ValueError`` on the root, naming ``what``, when a process has
        another value; returns ``value`` elsewhere.
        """
        if self.size == 1:
            return value
        values = self.comm.gather(value, root=0)
        if not self.is_root:
            return value
        differ = [rank for rank, other in enumerate(values) if not _same(other, values[0])]
        if differ:
            found = ", ".join(f"rank {rank}: {other!r}" for rank, other in enumerate(values))
            raise ValueError(
                f"{what} differs between the processes of its region ({found}); a field that "
                "is not Array is one value for the interface, the same on every process"
            )
        return values[0]


def _same(first: Any, second: Any) -> bool:
    """Whether two plain values are equal; NaN equals NaN."""
    if isinstance(first, float) and isinstance(second, float):
        return first == second or (first != first and second != second)
    return bool(first == second)
