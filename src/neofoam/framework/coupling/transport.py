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

Example::

    transport = InProcessTransport()
    transport.send("left", "right", "interface", {"T": np.ones(3)})
    transport.receive("left", "right", "interface")    # in right's thread
"""

from __future__ import annotations

import copy
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any

# What a side sends, by name: arrays with one entry per point of its patch, or one
# value for the whole interface.
Data = dict[str, Any]
# A coupled side: (region, patch, field).
SideKey = tuple[str, str, str]


class TransportClosed(RuntimeError):
    """A region waits for a message from a region that left the coupling."""


class Transport(ABC):
    """Messages between the regions of a coupled case, by ``(source, target, tag)``."""

    @abstractmethod
    def send(self, source: str, target: str, tag: str, message: Any) -> None:
        """Send ``message`` from region ``source`` to region ``target``; never blocks."""

    @abstractmethod
    def receive(self, source: str, target: str, tag: str) -> Any:
        """The next message ``source`` sent ``target`` under ``tag``; blocks until it is there.

        Raises :class:`TransportClosed` when ``source`` left (:meth:`close`) and
        no such message is waiting.
        """

    @abstractmethod
    def close(self, region: str) -> None:
        """Region ``region`` leaves the coupling (it finished or failed)."""


class InProcessTransport(Transport):
    """A transport between regions running in threads of one process: a mailbox per tag.

    Sends a copy, as a message would: a receiver that changes its data in place
    changes no one else's.
    """

    def __init__(self) -> None:
        self._messages: dict[tuple[str, str, str], deque[Any]] = defaultdict(deque)
        self._closed: set[str] = set()
        self._changed = threading.Condition()

    def send(self, source: str, target: str, tag: str, message: Any) -> None:
        with self._changed:
            self._messages[(source, target, tag)].append(copy.deepcopy(message))
            self._changed.notify_all()

    def receive(self, source: str, target: str, tag: str) -> Any:
        key = (source, target, tag)
        with self._changed:
            while not self._messages[key]:
                if source in self._closed:
                    raise TransportClosed(
                        f"region {target} waits for '{tag}' from region {source}, "
                        "which left the coupling"
                    )
                self._changed.wait()
            return self._messages[key].popleft()

    def close(self, region: str) -> None:
        with self._changed:
            self._closed.add(region)
            self._changed.notify_all()
