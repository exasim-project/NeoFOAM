# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The comm channel between regions on threads of one process: :class:`InProcessCommChannel`."""

from __future__ import annotations

import pickle
import threading
from typing import Any, Iterable, Optional

from .channel import DEFAULT_TIMEOUT, CouplingDeadlock, Mailbox, pickled


class InProcessCommChannel(Mailbox):
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
        super().__init__(timeout)
        self._deadlock: Optional[str] = None
        self._changed = threading.Condition()

    def start(self, regions: Iterable[str]) -> None:
        with self._changed:
            self._live |= set(regions) - self._closed

    def send(self, source: str, target: str, tag: str, message: Any) -> None:
        sent = pickle.loads(pickled(source, target, tag, message))
        with self._changed:
            self._messages[(source, target, tag)].append(sent)
            self._changed.notify_all()

    def receive(self, source: str, target: str, tag: str) -> Any:
        with self._changed:
            return self._take(source, target, tag)

    def _check_deadlock(self) -> None:
        if self._deadlock is not None:
            raise CouplingDeadlock(self._deadlock)

    def _wait(self, left: Optional[float]) -> None:
        if self._deadlocked():
            self._deadlock = self._report(
                "a coupling deadlock: every running region waits for a message no region sent"
            )
            self._changed.notify_all()
            raise CouplingDeadlock(self._deadlock)
        self._changed.wait(left)

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
