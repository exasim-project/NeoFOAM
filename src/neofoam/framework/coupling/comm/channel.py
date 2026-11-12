# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The :class:`CommChannel` between coupled regions, the only thing they share; its mailbox."""

from __future__ import annotations

import pickle
import time
from collections import defaultdict, deque
from typing import Any, Iterable, Literal, Optional, Protocol

# What a port sends of one data model, by field name: a row per entity, or a value.
Data = dict[str, Any]


# Seconds a receive of an InProcessCommChannel waits by default.
DEFAULT_TIMEOUT = 600.0


class Tag:
    """The tags of the coupling's messages, the one place that builds them."""

    TIME_DT = "time/dt"

    @staticmethod
    def data(exchange: str, key: str) -> str:
        """What a port sends of one data model (by its ``key``), each iteration."""
        return f"data:{exchange}/{key}"

    @staticmethod
    def port(exchange: str) -> str:
        """What a port sends and its geometry, told its partner once."""
        return f"port:{exchange}"

    @staticmethod
    def loop(loop: str, what: Literal["streams", "iteration", "decision"]) -> str:
        """A region's streams, its iteration, the leader's decision, in coupling loop ``loop``."""
        return f"{loop}/{what}"


class CommChannelError(RuntimeError):
    """A message between coupled regions cannot arrive."""


class CommChannelClosed(CommChannelError):
    """A region waits for a message from a region that left the coupling."""


class CouplingDeadlock(CommChannelError):
    """Every running region waits for a message no region will send."""


class CouplingTimeout(CommChannelError):
    """A region waited for a message longer than the comm channel's timeout."""


class CommChannel(Protocol):
    """Messages between the regions of a coupled case, by ``(source, target, tag)``.

    Point to point, as MPI: messages of one ``(source, target, tag)`` arrive in
    order; they are plain data (they pickle), their tags built by :class:`Tag`.
    """

    def send(self, source: str, target: str, tag: str, message: Any) -> None:
        """Send ``message`` from region ``source`` to region ``target``; never blocks."""

    def receive(self, source: str, target: str, tag: str) -> Any:
        """The next message ``source`` sent ``target`` under ``tag``; blocks until it is there.

        Raises :class:`CommChannelClosed` when ``source`` left and no such message
        waits, or another :class:`CommChannelError` when it cannot come.
        """

    def close(self, region: str) -> None:
        """Region ``region`` leaves the coupling (it finished or failed)."""

    def start(self, regions: Iterable[str]) -> None:
        """The regions ``regions`` run now, each until it leaves (:meth:`close`)."""

    def abort(self, regions: Iterable[str]) -> None:
        """A region failed: every region of ``regions`` leaves the coupling."""


class Mailbox:
    """The receive loop, its failure rules and its report, shared by the comm channels."""

    def __init__(self, timeout: Optional[float]) -> None:
        if timeout is not None and timeout < 0:
            raise ValueError(f"a comm channel timeout is at least 0 seconds, not {timeout}")
        self.timeout = timeout or None
        self._messages: dict[tuple[str, str, str], deque[Any]] = defaultdict(deque)
        self._closed: set[str] = set()
        self._aborted: set[str] = set()
        self._live: set[str] = set()
        # target -> (source, tag, since): the receive each region waits in
        self._waiting: dict[str, tuple[str, str, float]] = {}

    def _take(self, source: str, target: str, tag: str) -> Any:
        """The next message of ``(source, target, tag)``, waiting in :meth:`_wait`."""
        key = (source, target, tag)
        since = time.monotonic()
        self._waiting[target] = (source, tag, since)
        try:
            while not self._messages[key]:
                self._check_deadlock()
                if source in self._aborted:
                    raise CommChannelClosed(
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
                left = None if self.timeout is None else since + self.timeout - time.monotonic()
                if left is not None and left <= 0:
                    raise CouplingTimeout(
                        self._report(
                            f"a coupling timeout: region {target} waited {self.timeout:g} s "
                            f"for '{tag}' from region {source} (commChannelTimeout)"
                        )
                    )
                self._wait(left)
            return self._messages[key].popleft()
        finally:
            self._waiting.pop(target, None)

    def _check_deadlock(self) -> None:
        """Raise :class:`CouplingDeadlock` once one was told; only threads can tell."""

    def _wait(self, left: Optional[float]) -> None:
        """Wait at most ``left`` seconds (``None``: no end) for a message or a region leaving."""
        raise NotImplementedError

    def _report(self, headline: str) -> str:
        """``headline``, who waits for what since when, who left, the messages not received."""
        now = time.monotonic()
        lines = [headline + ":"]
        for target, (source, tag, since) in sorted(self._waiting.items()):
            lines.append(f"  {target} waits for '{tag}' from {source} (for {now - since:.1f} s)")
        for region in sorted(self._live - set(self._waiting)):
            lines.append(f"  {region} is busy (in no receive)")
        left = sorted(self._closed | self._aborted)
        if left:
            lines.append(f"left the coupling: {', '.join(left)}")
        undelivered = [
            f"{source}→{target} '{tag}'" + (f" ({len(queue)})" if len(queue) > 1 else "")
            for (source, target, tag), queue in sorted(self._messages.items())
            if queue
        ]
        lines.append(f"undelivered messages: {', '.join(undelivered) or 'none'}")
        return "\n".join(lines)


def pickled(source: str, target: str, tag: str, message: Any) -> bytes:
    """``message`` pickled; ``TypeError`` when it is not plain data."""
    try:
        return pickle.dumps(message, protocol=5)
    except Exception as error:
        raise TypeError(
            f"the message '{tag}' from region {source} to region {target} is not plain "
            f"data, it does not pickle: {error}"
        ) from error
