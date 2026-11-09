# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The MPI comm channel between region roots, and a region's processes (:class:`RegionComm`).

Without MPI, :class:`MpiTransport` runs over an in-memory channel between two
threads (``FakeChannel``: ``isend``, ``iprobe`` and ``recv`` of one tag, in
order per pair as MPI's), and :class:`RegionComm` of one process hands back
what it gets. Under ``mpirun`` ``mpi_transport_worker.py`` runs each scenario
on real processes: order and mailbox by tag, the timeout, a partner that closed
or aborted, pickling, and the gather and scatter of a region with uneven rows
and an empty process. Each ``mpirun`` runs in a subprocess with a timeout, so a
hang fails it; those tests need mpi4py and ``mpirun`` and skip without them.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest

from neofoam.framework.coupling import (
    CommChannelClosed,
    CouplingDeadlock,
    CouplingTimeout,
    MpiTransport,
)
from neofoam.framework.coupling.mpi import FollowerTransport, RegionComm

HERE = Path(__file__).parent
WORKER = HERE / "mpi_transport_worker.py"
TIMEOUT = 120


class _Request:
    def Test(self) -> bool:  # noqa: N802 - mpi4py's name
        return True

    def Wait(self) -> None:  # noqa: N802 - mpi4py's name
        pass


class FakeChannel:
    """One rank's end of an in-memory channel: what mpi4py's lowercase point to point does."""

    def __init__(self, rank: int, boxes: dict[int, deque[Any]], lock: threading.Lock) -> None:
        self.rank = rank
        self._boxes = boxes
        self._lock = lock

    def isend(self, payload: bytes, dest: int, tag: int) -> _Request:
        with self._lock:
            self._boxes[dest].append(payload)
        return _Request()

    def iprobe(self, tag: int) -> bool:
        with self._lock:
            return bool(self._boxes[self.rank])

    def recv(self, tag: int) -> bytes:
        with self._lock:
            payload: bytes = self._boxes[self.rank].popleft()
        return payload


def _pair(timeout: Optional[float] = 5.0) -> tuple[MpiTransport, MpiTransport]:
    boxes: dict[int, deque[Any]] = {0: deque(), 1: deque()}
    lock = threading.Lock()
    roots = {"a": 0, "b": 1}
    return (
        MpiTransport(FakeChannel(0, boxes, lock), roots, "a", timeout=timeout),
        MpiTransport(FakeChannel(1, boxes, lock), roots, "b", timeout=timeout),
    )


def _both(a: Callable[[], None], b: Callable[[], None]) -> None:
    """Run ``a`` and ``b`` on threads of their own; raise the first error."""
    errors: list[BaseException] = []

    def run(task: Callable[[], None]) -> None:
        try:
            task()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(task,)) for task in (a, b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "a comm channel hangs"
    if errors:
        raise errors[0]


def test_messages_arrive_by_tag_in_the_order_they_were_sent() -> None:
    a, b = _pair()
    for i in range(6):
        a.send("a", "b", "odd" if i % 2 else "even", {"i": i, "v": np.full(2, float(i))})

    odd = [b.receive("a", "b", "odd")["i"] for _ in range(3)]
    even = [b.receive("a", "b", "even")["i"] for _ in range(3)]

    assert odd == [1, 3, 5]
    assert even == [0, 2, 4]


def test_a_receive_times_out_naming_what_waits_and_what_arrived() -> None:
    a, b = _pair(timeout=0.05)
    a.send("a", "b", "extra", 1)
    a.send("a", "b", "extra", 2)

    with pytest.raises(CouplingTimeout) as error:
        b.receive("a", "b", "never")

    message = str(error.value)
    assert "region b waited 0.05 s for 'never' from region a (commChannelTimeout)" in message
    assert "undelivered messages: a→b 'extra' (2)" in message


def test_a_message_that_does_not_pickle_fails_at_its_send() -> None:
    a, _b = _pair()

    with pytest.raises(TypeError, match="the message 'f' from region a to region b is not plain"):
        a.send("a", "b", "f", lambda: 1)


def test_a_partner_that_closed_fails_a_receive_at_once_after_what_it_sent() -> None:
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

    _both(first, second)

    assert seen == ["bye"]


def test_a_partner_that_aborted_closes_the_comm_channel() -> None:
    a, b = _pair(timeout=None)
    a.abort(["a"])

    with pytest.raises(CommChannelClosed, match="which left the coupling"):
        b.receive("a", "b", "never")
    b.close("b")


def test_a_comm_channel_is_its_roots_alone() -> None:
    boxes: dict[int, deque[Any]] = {0: deque(), 1: deque()}
    channel = FakeChannel(1, boxes, threading.Lock())

    with pytest.raises(ValueError, match="only a region's root talks to the other regions"):
        MpiTransport(channel, {"a": 0, "b": 1}, "a")
    with pytest.raises(ValueError, match="at least 0 seconds"):
        MpiTransport(channel, {"a": 0, "b": 1}, "b", timeout=-1)
    comm_channel = MpiTransport(channel, {"a": 0, "b": 1}, "b")
    with pytest.raises(ValueError, match="the MPI comm channel of region b sends as region a"):
        comm_channel.send("a", "b", "x", 1)


def test_a_follower_talks_to_no_region() -> None:
    follower = FollowerTransport("left")

    with pytest.raises(RuntimeError, match="only the region's root talks to other regions"):
        follower.send("left", "right", "x", 1)
    with pytest.raises(RuntimeError, match="cannot receive 'x' from region right"):
        follower.receive("right", "left", "x")
    follower.close("left")


def test_a_region_on_one_process_hands_back_what_it_gets() -> None:
    comm = RegionComm()
    local = np.arange(6.0).reshape(2, 3)

    gathered = comm.gather(local)

    assert (comm.size, comm.rank, comm.is_root) == (1, 0, True)
    assert gathered is not None and gathered[0] is local and gathered[1] == (2,)
    assert comm.gather(None) == (None, None)
    assert comm.scatter(local, (2,)) is local
    assert comm.bcast("x") == "x"
    assert comm.agree("q", 1.0) == 1.0


def _mpirun() -> Optional[str]:
    """The ``mpirun`` next to the interpreter (its environment's MPI), else the one on PATH."""
    beside = Path(sys.executable).parent / "mpirun"
    return str(beside) if beside.exists() else shutil.which("mpirun")


needs_mpi = pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None or _mpirun() is None,
    reason="needs mpi4py and mpirun (the extra neofoam[mpi])",
)


@needs_mpi
@pytest.mark.parametrize(
    ("scenario", "processes"),
    [
        ("order", 2),
        ("timeout", 2),
        ("close", 2),
        ("abort", 2),
        ("pickle", 2),
        ("refusals", 2),
        ("region_comm", 5),
    ],
)
def test_mpi(scenario: str, processes: int) -> None:
    run = subprocess.run(
        [str(_mpirun()), "-np", str(processes), sys.executable, str(WORKER), scenario],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        env={**os.environ, "OMPI_MCA_rmaps_base_oversubscribe": "1"},
    )

    assert run.returncode == 0, f"stdout:\n{run.stdout[-4000:]}\nstderr:\n{run.stderr[-4000:]}"
    assert run.stdout.count(f"#OK {scenario}") == processes
