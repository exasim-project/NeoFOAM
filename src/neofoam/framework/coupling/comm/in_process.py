# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The comm channel between regions on threads of one process: :class:`InProcessCommChannel`."""

from __future__ import annotations

import pickle
import threading
import time
from collections import defaultdict, deque
from typing import Any, Iterable, Optional

from .channel import DEFAULT_TIMEOUT, CommChannelClosed, CouplingDeadlock, CouplingTimeout


class InProcessCommChannel:
    """A comm channel between regions running in threads of one process: a mailbox per tag.

    Messages are pickled, as between processes: the receiver gets its own copy.
    Once :meth:`start` named the running regions, a receive raises
    :class:`CouplingDeadlock` as soon as every running region waits in vain, and
    :class:`CouplingTimeout` after ``timeout`` seconds (``None`` or ``0``: never).

    Example::

        comm_channel = InProcessCommChannel()
        comm_channel.send("left", "right", "interface", {"T": np.ones(3)})
        comm_channel.receive("left", "right", "interface")    # in right's thread
    """

    def __init__(self, timeout: Optional[float] = DEFAULT_TIMEOUT) -> None:
        if timeout is not None and timeout < 0:
            raise ValueError(f"a comm channel timeout is at least 0 seconds, not {timeout}")
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
                                f"for '{tag}' from region {source} (commChannelTimeout)"
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
