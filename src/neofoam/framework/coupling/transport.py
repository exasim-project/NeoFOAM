# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""How data travels between coupled regions: the :class:`Transport`, the only thing they share.

Point to point, as MPI: a send never blocks, a receive blocks until the message
is there; messages of one ``(source, target, tag)`` arrive in order. Messages are
plain data (they pickle), their tags built by :class:`Tag` alone.

Example::

    transport = InProcessTransport()
    transport.send("left", "right", "interface", {"T": np.ones(3)})
    transport.receive("left", "right", "interface")    # in right's thread
"""

from __future__ import annotations

import pickle
import threading
import time
from collections import defaultdict, deque
from typing import Any, Iterable, Literal, Optional, Protocol

# What a side sends, by field name: one value (or row) per point of its patch.
Data = dict[str, Any]

# Seconds a receive of an InProcessTransport waits by default.
DEFAULT_TIMEOUT = 600.0


class Tag:
    """The tags of the coupling's messages, the one place that builds them."""

    TIME_DT = "time/dt"

    @staticmethod
    def data(exchange: str) -> str:
        """What a side sends, each pass."""
        return f"data:{exchange}"

    @staticmethod
    def side(exchange: str) -> str:
        """A side's models and points, sent to its partner once."""
        return f"side:{exchange}"

    @staticmethod
    def node(node: str, what: Literal["sides", "pass", "decision"]) -> str:
        """A region's sides, its pass and the leader's decision, within coupling node ``node``."""
        return f"{node}/{what}"


class CouplingTransportError(RuntimeError):
    """A message between coupled regions cannot arrive."""


class TransportClosed(CouplingTransportError):
    """A region waits for a message from a region that left the coupling."""


class CouplingDeadlock(CouplingTransportError):
    """Every running region waits for a message no region will send."""


class CouplingTimeout(CouplingTransportError):
    """A region waited for a message longer than the transport's timeout."""


class Transport(Protocol):
    """Messages between the regions of a coupled case, by ``(source, target, tag)``."""

    def send(self, source: str, target: str, tag: str, message: Any) -> None:
        """Send ``message`` from region ``source`` to region ``target``; never blocks."""

    def receive(self, source: str, target: str, tag: str) -> Any:
        """The next message ``source`` sent ``target`` under ``tag``; blocks until it is there.

        Raises :class:`TransportClosed` when ``source`` left and no such message
        waits, or another :class:`CouplingTransportError` when it cannot come.
        """

    def close(self, region: str) -> None:
        """Region ``region`` leaves the coupling (it finished or failed)."""

    def start(self, regions: Iterable[str]) -> None:
        """The regions ``regions`` run now, each until it leaves (:meth:`close`)."""

    def abort(self, regions: Iterable[str]) -> None:
        """A region failed: every region of ``regions`` leaves the coupling."""


class InProcessTransport:
    """A transport between regions running in threads of one process: a mailbox per tag.

    Each message is pickled and unpickled, as between processes: the receiver
    gets its own copy, and a message that is not plain data raises ``TypeError``.
    Once :meth:`start` named the running regions, a receive raises
    :class:`CouplingDeadlock` as soon as every running region waits for a message
    that is not there, or waits for a region that finished; it raises
    :class:`TransportClosed` for a region that left by :meth:`abort`, and
    :class:`CouplingTimeout` after ``timeout`` seconds (``None`` or ``0``: no end).
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
        sent = self._copy(source, target, tag, message)
        with self._changed:
            self._messages[(source, target, tag)].append(sent)
            self._changed.notify_all()

    def _copy(self, source: str, target: str, tag: str, message: Any) -> Any:
        """The copy of ``message`` that arrives: a pickle round trip."""
        try:
            return pickle.loads(pickle.dumps(message))
        except Exception as error:
            raise TypeError(
                f"the message '{tag}' from region {source} to region {target} is not plain "
                f"data, it does not pickle: {error}"
            ) from error

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
