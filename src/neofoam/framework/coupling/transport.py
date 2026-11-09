# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""How data travels between coupled regions: the :class:`Transport`, the only thing they share.

Point to point, as MPI: a region sends a message to another region under a tag
and the other receives it. A send never blocks; a receive blocks until the
message is there. Messages of one ``(source, target, tag)`` arrive in the order
they were sent. Messages are plain data (dicts of numpy arrays and numbers,
points), so a transport between processes can pickle them.

:class:`InProcessTransport` connects regions that run in threads of one process.
A transport between processes (e.g. MPI, one rank group per region) implements
the same three methods; boundary conditions and solvers do not see it.

:class:`InProcessTransport` raises :class:`CouplingDeadlock` at once when every
region still running waits in :meth:`~Transport.receive` for a message none of
them sent, or when a region waits for one that finished; as a fallback it raises
:class:`CouplingTimeout` when a receive waits longer than ``timeout`` seconds.
Both carry a report: who waits for what, and the messages sent but not received.

Example::

    transport = InProcessTransport()
    transport.send("left", "right", "interface", {"T": np.ones(3)})
    transport.receive("left", "right", "interface")    # in right's thread
"""

from __future__ import annotations

import copy
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Iterable, Optional

# What a side sends, by name: arrays with one entry per point of its patch, or one
# value for the whole interface.
Data = dict[str, Any]
# A coupled side: (region, patch, field).
SideKey = tuple[str, str, str]

# How long a receive of an InProcessTransport waits by default, in seconds: 10 minutes.
DEFAULT_TIMEOUT = 600.0


class CouplingTransportError(RuntimeError):
    """A message between coupled regions cannot arrive."""


class TransportClosed(CouplingTransportError):
    """A region waits for a message from a region that left the coupling."""


class CouplingDeadlock(CouplingTransportError):
    """Every running region waits for a message no region will send."""


class CouplingTimeout(CouplingTransportError):
    """A region waited for a message longer than the transport's timeout."""


class Transport(ABC):
    """Messages between the regions of a coupled case, by ``(source, target, tag)``."""

    @abstractmethod
    def send(self, source: str, target: str, tag: str, message: Any) -> None:
        """Send ``message`` from region ``source`` to region ``target``; never blocks."""

    @abstractmethod
    def receive(self, source: str, target: str, tag: str) -> Any:
        """The next message ``source`` sent ``target`` under ``tag``; blocks until it is there.

        Raises :class:`TransportClosed` when ``source`` left (:meth:`close`) and
        no such message is waiting, or another :class:`CouplingTransportError`
        when the message cannot come (:class:`InProcessTransport`).
        """

    @abstractmethod
    def close(self, region: str) -> None:
        """Region ``region`` leaves the coupling (it finished or failed)."""

    def start(self, regions: Iterable[str]) -> None:
        """The regions ``regions`` run now, each until it leaves (:meth:`close`).

        A transport may watch them for a deadlock; by default it does nothing.
        """

    def abort(self, regions: Iterable[str]) -> None:
        """A region failed: every region of ``regions`` leaves the coupling (:meth:`close`)."""
        for region in regions:
            self.close(region)


class InProcessTransport(Transport):
    """A transport between regions running in threads of one process: a mailbox per tag.

    Sends a copy, as a message would: a receiver that changes its data in place
    changes no one else's.

    Once :meth:`start` named the running regions (as
    :func:`~neofoam.framework.coupling.run_regions` does), a receive raises
    :class:`CouplingDeadlock` as soon as every running region waits in a
    receive and none of the messages they wait for is there: only these
    regions send. A receive from a region that finished (left by :meth:`close`,
    not by :meth:`abort`) raises :class:`CouplingDeadlock` too; one from a
    region that left by :meth:`abort` raises :class:`TransportClosed`. A
    receive that waits longer than ``timeout`` seconds raises
    :class:`CouplingTimeout`; ``timeout`` ``None`` or ``0`` waits without end.
    Raises ``ValueError`` for a negative ``timeout``.
    """

    def __init__(self, timeout: Optional[float] = DEFAULT_TIMEOUT) -> None:
        if timeout is not None and timeout < 0:
            raise ValueError(f"a transport timeout is at least 0 seconds, not {timeout}")
        self.timeout = timeout or None
        self._messages: dict[tuple[str, str, str], deque[Any]] = defaultdict(deque)
        self._closed: set[str] = set()
        self._aborted: set[str] = set()
        self._live: set[str] = set()
        # target -> (source, tag, since): the receive each region waits in
        self._waiting: dict[str, tuple[str, str, float]] = {}
        self._deadlock: Optional[str] = None
        self._changed = threading.Condition()

    def start(self, regions: Iterable[str]) -> None:
        with self._changed:
            self._live |= set(regions) - self._closed

    def send(self, source: str, target: str, tag: str, message: Any) -> None:
        with self._changed:
            self._messages[(source, target, tag)].append(copy.deepcopy(message))
            self._changed.notify_all()

    def receive(self, source: str, target: str, tag: str) -> Any:
        key = (source, target, tag)
        with self._changed:
            since = time.monotonic()
            self._waiting[target] = (source, tag, since)
            try:
                while not self._messages[key]:
                    if self._deadlock is not None:
                        raise CouplingDeadlock(self._deadlock)
                    if source in self._aborted:
                        raise TransportClosed(
                            f"region {target} waits for '{tag}' from region {source}, "
                            "which left the coupling"
                        )
                    if source in self._closed:
                        raise CouplingDeadlock(
                            self._report(
                                f"a coupling deadlock: region {target} waits for '{tag}' from "
                                f"region {source}, which finished"
                            )
                        )
                    if self._deadlocked():
                        self._deadlock = self._report(
                            "a coupling deadlock: every running region waits for a message "
                            "no region sent"
                        )
                        self._changed.notify_all()
                        raise CouplingDeadlock(self._deadlock)
                    if self.timeout is None:
                        self._changed.wait()
                        continue
                    left = since + self.timeout - time.monotonic()
                    if left <= 0:
                        raise CouplingTimeout(
                            self._report(
                                f"a coupling timeout: region {target} waited {self.timeout:g} s "
                                f"for '{tag}' from region {source} (transportTimeout)"
                            )
                        )
                    self._changed.wait(left)
                return self._messages[key].popleft()
            finally:
                self._waiting.pop(target, None)

    def close(self, region: str) -> None:
        with self._changed:
            self._leave([region])

    def abort(self, regions: Iterable[str]) -> None:
        with self._changed:
            regions = [region for region in regions if region not in self._closed]
            self._aborted |= set(regions)
            self._leave(regions)

    def _leave(self, regions: Iterable[str]) -> None:
        """``regions`` leave; the caller holds the lock."""
        for region in regions:
            self._closed.add(region)
            self._live.discard(region)
        self._changed.notify_all()

    def _deadlocked(self) -> bool:
        """Whether every running region waits for a message that is not there."""
        return bool(self._live) and all(
            region in self._waiting
            and not self._messages[(self._waiting[region][0], region, self._waiting[region][1])]
            for region in self._live
        )

    def _report(self, headline: str) -> str:
        """``headline``, then who waits for what since when and the messages not received."""
        now = time.monotonic()
        lines = [headline + ":"]
        for target, (source, tag, since) in sorted(self._waiting.items()):
            lines.append(f"  {target} waits for '{tag}' from {source} (for {now - since:.1f} s)")
        for region in sorted(self._live - set(self._waiting)):
            lines.append(f"  {region} is busy (in no receive)")
        undelivered = [
            f"{source}→{target} '{tag}'" + (f" ({len(queue)})" if len(queue) > 1 else "")
            for (source, target, tag), queue in sorted(self._messages.items())
            if queue
        ]
        lines.append(f"undelivered messages: {', '.join(undelivered) or 'none'}")
        return "\n".join(lines)
