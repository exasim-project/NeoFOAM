# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The :class:`CommChannel` between coupled regions, the only thing they share."""

from __future__ import annotations

from typing import Any, Iterable, Literal, Protocol

# What a port sends of one data model, by field name: a row per entity, or a value.
Data = dict[str, Any]


# Seconds a receive of an InProcessCommChannel waits by default.
DEFAULT_TIMEOUT = 600.0


class Tag:
    """The tags of the coupling's messages, the one place that builds them."""

    TIME_DT = "time/dt"

    @staticmethod
    def data(exchange: str, key: str) -> str:
        """What a port sends of one data model (by its ``key``), each pass."""
        return f"data:{exchange}/{key}"

    @staticmethod
    def side(exchange: str) -> str:
        """What a port sends and its geometry, told its partner once."""
        return f"side:{exchange}"

    @staticmethod
    def node(node: str, what: Literal["sides", "pass", "decision"]) -> str:
        """A region's streams, its pass and the leader's decision, within coupling node ``node``."""
        return f"{node}/{what}"


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
