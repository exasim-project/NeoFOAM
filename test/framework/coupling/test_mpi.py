# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The MPI comm channel between region roots, and a region's processes (:class:`RegionComm`).

Without MPI the comm channel runs over an in-memory communicator (``FakeComm``, in order per
pair as MPI); under ``mpirun`` ``mpi_comm_channel_worker.py`` runs each scenario on real
processes, in a subprocess with a timeout.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest
from _mpirun import mpirun, needs_mpi

from neofoam.framework.coupling import (
    CommChannelClosed,
    CouplingDeadlock,
    CouplingTimeout,
    MpiCommChannel,
)

WORKER = Path(__file__).parent / "mpi_comm_channel_worker.py"
# seconds: a hang guard, never reached by a passing test
HANG = 30.0


class _Request:
    def __init__(self, completes: bool) -> None:
        self._completes = completes

    def Test(self) -> bool:  # noqa: N802 - mpi4py's name
        return self._completes


class FakeComm:
    """One rank's end of an in-memory communicator; ``completes=False``: no send ever completes."""

    def __init__(
        self, rank: int, boxes: dict[int, deque[Any]], lock: threading.Lock, completes: bool
    ) -> None:
        self.rank = rank
        self._boxes = boxes
        self._lock = lock
        self._completes = completes

    def isend(self, payload: bytes, dest: int, tag: int) -> _Request:
        with self._lock:
            self._boxes[dest].append(payload)
        return _Request(self._completes)

    def iprobe(self, tag: int) -> bool:
        with self._lock:
            return bool(self._boxes[self.rank])

    def recv(self, tag: int) -> bytes:
        with self._lock:
            payload: bytes = self._boxes[self.rank].popleft()
        return payload


def _pair(
    timeout: Optional[float] = HANG, completes: bool = True
) -> tuple[MpiCommChannel, MpiCommChannel]:
    boxes: dict[int, deque[Any]] = {0: deque(), 1: deque()}
    lock = threading.Lock()
    roots = {"a": 0, "b": 1}
    return (
        MpiCommChannel(FakeComm(0, boxes, lock, completes), roots, "a", timeout=timeout),
        MpiCommChannel(FakeComm(1, boxes, lock, completes), roots, "b", timeout=timeout),
    )


def _within(*tasks: Callable[[], None]) -> None:
    """Run ``tasks`` on threads of their own; fail if one hangs, raise the first error."""
    errors: list[BaseException] = []

    def run(task: Callable[[], None]) -> None:
        try:
            task()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(task,), daemon=True) for task in tasks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=HANG)
    assert not any(thread.is_alive() for thread in threads), "a comm channel call hangs"
    if errors:
        raise errors[0]


def test_mpi_comm_channel_delivers_by_tag_in_the_order_sent() -> None:
    a, b = _pair()
    for i in range(6):
        a.send("a", "b", "odd" if i % 2 else "even", {"i": i, "v": np.full(2, float(i))})

    odd = [b.receive("a", "b", "odd")["i"] for _ in range(3)]
    even = [b.receive("a", "b", "even")["i"] for _ in range(3)]

    assert odd == [1, 3, 5]
    assert even == [0, 2, 4]


def test_mpi_comm_channel_receive_times_out_naming_the_messages_not_received() -> None:
    a, b = _pair(timeout=0.05)
    a.send("a", "b", "extra", 1)
    a.send("a", "b", "extra", 2)

    with pytest.raises(CouplingTimeout) as error:
        b.receive("a", "b", "never")

    message = str(error.value)
    assert "region b waited 0.05 s for 'never' from region a (commChannelTimeout)" in message
    assert "undelivered messages: a→b 'extra' (2)" in message


def test_mpi_comm_channel_refuses_a_message_that_does_not_pickle() -> None:
    a, _b = _pair()

    with pytest.raises(TypeError, match="the message 'f' from region a to region b is not plain"):
        a.send("a", "b", "f", lambda: 1)


def test_mpi_comm_channel_receive_from_a_closed_partner_raises_after_what_it_sent() -> None:
    a, b = _pair(timeout=None)
    seen: list[Any] = []

    def first() -> None:
        a.send("a", "b", "last", "bye")
        a.close("a")

    def second() -> None:
        seen.append(b.receive("a", "b", "last"))
        with pytest.raises(CouplingDeadlock, match="from region a, which finished"):
            b.receive("a", "b", "never")
        b.close("b")

    _within(first, second)

    assert seen == ["bye"]


def test_mpi_comm_channel_receive_from_an_aborted_partner_raises_comm_channel_closed() -> None:
    a, b = _pair(timeout=None)
    a.abort(["a"])

    with pytest.raises(CommChannelClosed, match="which left the coupling"):
        b.receive("a", "b", "never")
    _within(lambda: b.close("b"))


def test_mpi_comm_channel_close_times_out_naming_the_roots_that_did_not_close() -> None:
    a, _b = _pair(timeout=0.05)

    with pytest.raises(CouplingTimeout) as error:
        _within(lambda: a.close("a"))

    message = str(error.value)
    assert "region a waited 0.05 s to close (commChannelTimeout); not closed: b" in message
    assert "sends not complete: none" in message


def test_mpi_comm_channel_close_times_out_naming_the_sends_not_complete() -> None:
    a, b = _pair(timeout=0.05, completes=False)
    a.send("a", "b", "x", 1)
    b.abort(["b"])

    with pytest.raises(CouplingTimeout) as error:
        _within(lambda: a.close("a"))

    assert "not closed: none; sends not complete: a→b 'x', a→b 'closed'" in str(error.value)


def test_mpi_comm_channel_belongs_to_its_root() -> None:
    boxes: dict[int, deque[Any]] = {0: deque(), 1: deque()}
    comm = FakeComm(1, boxes, threading.Lock(), completes=True)

    with pytest.raises(ValueError, match="only a region's root talks to the other regions"):
        MpiCommChannel(comm, {"a": 0, "b": 1}, "a")
    with pytest.raises(ValueError, match="at least 0 seconds"):
        MpiCommChannel(comm, {"a": 0, "b": 1}, "b", timeout=-1)
    comm_channel = MpiCommChannel(comm, {"a": 0, "b": 1}, "b")
    with pytest.raises(ValueError, match="the MPI comm channel of region b sends as region a"):
        comm_channel.send("a", "b", "x", 1)


@needs_mpi
@pytest.mark.parametrize(
    ("scenario", "processes"),
    [
        ("order", 2),
        ("timeout", 2),
        ("close", 2),
        ("close_timeout", 2),
        ("abort", 2),
        ("pickle", 2),
        ("refusals", 2),
        ("region_comm", 5),
    ],
)
def test_mpi_comm_channel_under_mpirun(scenario: str, processes: int) -> None:
    run = subprocess.run(
        [str(mpirun()), "-np", str(processes), sys.executable, str(WORKER), scenario],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "OMPI_MCA_rmaps_base_oversubscribe": "1"},
    )

    assert run.returncode == 0, f"stdout:\n{run.stdout[-4000:]}\nstderr:\n{run.stderr[-4000:]}"
    assert run.stdout.count(f"#OK {scenario}") == processes
