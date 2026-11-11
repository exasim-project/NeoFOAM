# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run one scenario of :mod:`neofoam.framework.coupling.mpi` under ``mpirun``, print ``#OK``.

Rank 0 is the root of region ``a``, rank 1 of ``b``; ``region_comm`` has two
regions on ranks 0-1 and 2-4. A failed check aborts every rank.
"""

import sys
import traceback
from typing import Any

import numpy as np
from mpi4py import MPI

from neofoam.framework.coupling import (
    Array,
    CommChannelClosed,
    CouplingDeadlock,
    CouplingTimeout,
    FaceData,
    Geometry,
    Global,
    MpiCommChannel,
    Patch,
    Scalar,
    ValueData,
)
from neofoam.framework.coupling.coupled import CoupledPort
from neofoam.framework.coupling.data import Stream
from neofoam.framework.coupling.mpi import RegionComm, _Port

WORLD = MPI.COMM_WORLD
ROOTS = {"a": 0, "b": 1}


def _comm_channel(timeout: float = 30.0) -> tuple[MpiCommChannel, str]:
    region = "a" if WORLD.rank == 0 else "b"
    return MpiCommChannel(WORLD.Dup(), ROOTS, region, timeout=timeout), region


def _raises(error: type[BaseException], call: Any) -> str:
    try:
        call()
    except error as raised:
        return str(raised)
    raise AssertionError(f"expected {error.__name__}")


def order() -> None:
    """Messages of two tags interleaved arrive by tag, each tag in the order sent."""
    comm_channel, region = _comm_channel()
    if region == "a":
        for i in range(40):
            comm_channel.send(
                "a", "b", "odd" if i % 2 else "even", {"i": i, "v": np.full(i, i * 0.5)}
            )
        comm_channel.send("a", "b", "done", None)
        assert comm_channel.receive("b", "a", "back") == "thanks"
    else:
        odd = [comm_channel.receive("a", "b", "odd") for _ in range(20)]
        assert comm_channel.receive("a", "b", "done") is None
        even = [comm_channel.receive("a", "b", "even") for _ in range(20)]
        assert [m["i"] for m in odd] == list(range(1, 40, 2))
        assert [m["i"] for m in even] == list(range(0, 40, 2))
        for m in odd + even:
            np.testing.assert_array_equal(m["v"], np.full(m["i"], m["i"] * 0.5))
        comm_channel.send("b", "a", "back", "thanks")
    comm_channel.close(region)


def timeout() -> None:
    """A receive times out, naming what it waits for and what arrived but was not received."""
    comm_channel, region = _comm_channel(timeout=0.5)
    if region == "a":
        comm_channel.send("a", "b", "extra", 1)
        comm_channel.send("a", "b", "extra", 2)
        comm_channel.send("a", "b", "other", 3)
        comm_channel.timeout = None
        assert comm_channel.receive("b", "a", "seen") is True
    else:
        message = _raises(CouplingTimeout, lambda: comm_channel.receive("a", "b", "never"))
        assert "region b waited 0.5 s for 'never' from region a (commChannelTimeout)" in message
        assert "b waits for 'never' from a" in message
        assert "undelivered messages: a→b 'extra' (2), a→b 'other'" in message
        comm_channel.send("b", "a", "seen", True)
    comm_channel.close(region)


def close() -> None:
    """A partner that closed: what it sent before still arrives, then a receive fails at once."""
    comm_channel, region = _comm_channel(timeout=30.0)
    if region == "a":
        comm_channel.send("a", "b", "last", "bye")
        comm_channel.close("a")
    else:
        assert comm_channel.receive("a", "b", "last") == "bye"
        message = _raises(CouplingDeadlock, lambda: comm_channel.receive("a", "b", "never"))
        assert "region b waits for 'never' from region a, which finished" in message
        comm_channel.close("b")


def close_timeout() -> None:
    """A partner that does not close: close gives up after the timeout, naming it."""
    comm_channel, region = _comm_channel(timeout=0.5)
    if region == "a":
        message = _raises(CouplingTimeout, lambda: comm_channel.close("a"))
        assert "region a waited 0.5 s to close (commChannelTimeout); not closed: b" in message
        WORLD.Barrier()
    else:
        WORLD.Barrier()
        comm_channel.close("b")


def abort() -> None:
    """A partner that aborted: a receive fails at once with CommChannelClosed."""
    comm_channel, region = _comm_channel(timeout=30.0)
    if region == "a":
        comm_channel.send("a", "b", "last", "bye")
        comm_channel.abort(["a"])
        comm_channel.close("a")  # it left already: nothing more to tell
    else:
        assert comm_channel.receive("a", "b", "last") == "bye"
        message = _raises(CommChannelClosed, lambda: comm_channel.receive("a", "b", "never"))
        assert "which left the coupling" in message
        comm_channel.close("b")


def pickle() -> None:
    """Plain data pickles across; a message that does not pickle fails at its send."""
    comm_channel, region = _comm_channel()
    payload = {"T": np.arange(6.0).reshape(2, 3), "n": (1, None, "x"), "inf": float("inf")}
    if region == "a":
        message = _raises(TypeError, lambda: comm_channel.send("a", "b", "f", lambda: 1))
        assert "the message 'f' from region a to region b is not plain data" in message
        comm_channel.send("a", "b", "data", payload)
    else:
        got = comm_channel.receive("a", "b", "data")
        np.testing.assert_array_equal(got["T"], payload["T"])
        assert got["n"] == payload["n"] and got["inf"] == payload["inf"]
    comm_channel.close(region)


def refusals() -> None:
    """A comm channel is its root's: another rank, or another region's send, is refused."""
    other = "b" if WORLD.rank == 0 else "a"
    message = _raises(ValueError, lambda: MpiCommChannel(WORLD, ROOTS, other))
    assert "only a region's root talks to the other regions" in message
    comm_channel, region = _comm_channel()
    message = _raises(ValueError, lambda: comm_channel.send(other, region, "x", 1))
    assert f"the MPI comm channel of region {region} sends as region {other}" in message
    comm_channel.close(region)


def region_comm() -> None:
    """Gather and scatter by rows: uneven counts, an empty process (a root too)."""
    color = 0 if WORLD.rank < 2 else 1
    comm = RegionComm(WORLD.Split(color, key=WORLD.rank))
    # rows per process: region 0 [3, 0], region 1 [0, 2, 4]; an empty one as shape (0,)
    rows = {0: 3, 1: 0, 2: 0, 3: 2, 4: 4}[WORLD.rank]
    local = (WORLD.rank * 100 + np.arange(rows * 3.0)).reshape(rows, 3) if rows else np.empty(0)
    gathered = comm.gather(local)
    expected_counts = {0: (3, 0), 1: (0, 2, 4)}[color]
    if comm.is_root:
        assert gathered is not None
        full, counts = gathered
        assert counts == expected_counts, counts
        assert full.shape == (sum(expected_counts), 3), full.shape
        first = WORLD.rank
        parts = [
            (rank * 100 + np.arange(n * 3.0)).reshape(n, 3)
            for rank, n in zip(range(first, first + len(expected_counts)), expected_counts)
            if n
        ]
        np.testing.assert_array_equal(full, np.concatenate(parts))
        back = comm.scatter(2 * full, counts)
    else:
        assert gathered is None
        back = comm.scatter(None, None)
    assert len(back) == rows
    if rows:
        np.testing.assert_array_equal(back, 2 * local)
    flat = comm.gather(np.arange(float(rows)))
    if comm.is_root:
        assert flat is not None and flat[1] == expected_counts
    assert comm.bcast(("decision", color) if comm.is_root else None) == ("decision", color)
    nothing = comm.gather(None)
    assert nothing == ((None, None) if comm.is_root else None)
    if comm.is_root:
        message = _raises(ValueError, lambda: comm.gather(np.zeros(1)))
        assert "have no data, the others have" in message
    else:
        comm.gather(None)
    if comm.is_root:
        message = _raises(ValueError, lambda: comm.scatter(np.zeros(3), (1, 1)))
        assert "cannot scatter 3 rows" in message


class _Flux(FaceData):
    q: Array


class _Heat(ValueData):
    q: Scalar


def ports() -> None:
    """A port over 3 processes: a patch gathers rows and scatters them back; a Global the root's."""
    comm = RegionComm(WORLD)
    rows = WORLD.rank + 1  # 1, 2, 3 faces
    received: list[Any] = []
    patch = CoupledPort(
        "a", "top", Patch("top"), (Stream(_Flux, print),), (Stream(_Flux, received.append),)
    )
    points = np.full((rows, 3), float(WORLD.rank))
    part = _Port(comm, patch, Geometry.of(points, np.ones(rows)))
    if comm.is_root:
        points = np.asarray(part.geometry.points)
        assert part.geometry.size == 6 and points[-1, 0] == 2.0, part.geometry
    whole: Any = part.gather_sent(Stream(_Flux, print), _Flux(q=np.full(rows, float(WORLD.rank))))
    if comm.is_root:
        assert whole.q.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 2.0], whole.q
    part.hand_over({"q": _Flux(q=10.0 * whole.q)} if comm.is_root else {})
    assert received[-1].q.tolist() == [10.0 * WORLD.rank] * rows, received[-1].q
    value = CoupledPort("a", "Q", Global("Q"), (), (Stream(_Heat, received.append),))
    part = _Port(comm, value, Geometry(size=1))
    root_value: Any = part.gather_sent(Stream(_Heat, print), _Heat(q=WORLD.rank))
    assert root_value.q == WORLD.rank
    part.hand_over({"q": _Heat(q=7.0)} if comm.is_root else {})
    assert received[-1].q == 7.0, received[-1]


SCENARIOS = {
    f.__name__: f
    for f in (order, timeout, close, close_timeout, abort, pickle, refusals, region_comm, ports)
}


if __name__ == "__main__":
    scenario = sys.argv[1]
    try:
        SCENARIOS[scenario]()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        WORLD.Abort(1)
    WORLD.Barrier()
    print(f"#OK {scenario}", flush=True)
