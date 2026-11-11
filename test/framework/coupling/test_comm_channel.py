# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The in-process comm channel tells a deadlock at once and times out as a fallback.

Regions ``a``, ``b`` (and ``c``) are plain tasks run by :func:`run_regions`,
each receiving from or sending to the others on the comm channel directly.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
from framework.coupling.regions import TIMEOUT

from neofoam.framework.coupling import (
    CommChannelClosed,
    CommChannelError,
    CouplingDeadlock,
    CouplingTimeout,
    InProcessCommChannel,
)
from neofoam.framework.coupling.comm_channel import Tag
from neofoam.framework.coupling.region import run_regions


def _until_waiting(comm_channel: InProcessCommChannel, region: str) -> None:
    """Block until ``region`` waits in a receive of ``comm_channel`` (which notifies no one)."""
    with comm_channel._changed:
        while region not in comm_channel._waiting:
            comm_channel._changed.wait(0.001)


def _waits(lines: str) -> list[str]:
    """The lines of a report that tell a region's state, without their durations."""
    return [line.split(" (for")[0].strip() for line in lines.splitlines() if line.startswith("  ")]


def _errors(
    comm_channel: InProcessCommChannel, tasks: dict[str, Callable[[], Any]]
) -> dict[str, Any]:
    """Run ``tasks`` on their own threads as run_regions does; the error each raised, by region."""
    errors: dict[str, Any] = {}

    def run(region: str, task: Callable[[], Any]) -> None:
        try:
            task()
        except CommChannelError as error:
            errors[region] = error
        finally:
            comm_channel.close(region)

    threads = [threading.Thread(target=run, args=item) for item in tasks.items()]
    comm_channel.start(tasks)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_two_regions_receiving_from_each_other_raise_a_deadlock_at_once() -> None:
    comm_channel = InProcessCommChannel(timeout=TIMEOUT)

    errors = _errors(
        comm_channel,
        {
            "a": lambda: comm_channel.receive("b", "a", "x"),
            "b": lambda: comm_channel.receive("a", "b", "y"),
        },
    )

    assert set(errors) == {"a", "b"}
    assert all(isinstance(error, CouplingDeadlock) for error in errors.values())
    report = str(errors["a"])
    assert report.startswith("a coupling deadlock: every running region waits")
    assert _waits(report) == ["a waits for 'x' from b", "b waits for 'y' from a"]
    assert report.endswith("undelivered messages: none")


def test_a_deadlock_lists_the_messages_sent_but_not_received() -> None:
    comm_channel = InProcessCommChannel()

    def a() -> None:
        comm_channel.send("a", "b", "z", 1.0)
        comm_channel.send("a", "b", "z", 2.0)
        comm_channel.receive("b", "a", "x")

    errors = _errors(comm_channel, {"a": a, "b": lambda: comm_channel.receive("a", "b", "y")})

    assert str(errors["b"]).endswith("undelivered messages: a→b 'z' (2)")


def test_waiting_for_a_region_that_finished_raises_a_deadlock_at_once() -> None:
    comm_channel = InProcessCommChannel()
    comm_channel.start(["a", "b"])
    comm_channel.close("b")

    with pytest.raises(
        CouplingDeadlock, match="region a waits for 'x' from region b, which finished"
    ):
        comm_channel.receive("b", "a", "x")


def test_a_message_a_finished_region_sent_is_still_received() -> None:
    comm_channel = InProcessCommChannel()
    comm_channel.start(["a", "b"])
    comm_channel.send("b", "a", "x", 1.0)
    comm_channel.close("b")

    assert comm_channel.receive("b", "a", "x") == 1.0


def test_waiting_for_a_region_that_failed_raises_comm_channel_closed() -> None:
    comm_channel = InProcessCommChannel()
    comm_channel.start(["a", "b"])
    comm_channel.close("b")  # finished first: aborting leaves it finished
    comm_channel.abort(["a", "b", "c"])

    with pytest.raises(CommChannelClosed, match="region a waits for 'y' from region c"):
        comm_channel.receive("c", "a", "y")
    with pytest.raises(CouplingDeadlock, match="which finished"):
        comm_channel.receive("b", "a", "x")


def test_a_region_that_finished_is_neither_running_nor_waiting() -> None:
    comm_channel = InProcessCommChannel(timeout=TIMEOUT)
    finished = threading.Event()

    def a() -> None:
        finished.wait()
        comm_channel.receive("b", "a", "x")

    def c() -> None:
        comm_channel.close("c")
        finished.set()

    errors = _errors(
        comm_channel, {"a": a, "b": lambda: comm_channel.receive("a", "b", "y"), "c": c}
    )

    assert _waits(str(errors["a"])) == ["a waits for 'x' from b", "b waits for 'y' from a"]


def test_a_region_waiting_while_its_partner_computes_waits() -> None:
    comm_channel = InProcessCommChannel(timeout=TIMEOUT)
    received: list[float] = []

    def b() -> None:
        _until_waiting(comm_channel, "a")
        comm_channel.send("b", "a", "x", 1.0)

    run_regions(
        {
            "a": lambda: received.append(comm_channel.receive("b", "a", "x")),
            "b": b,
        },
        comm_channel,
    )

    assert received == [1.0]


def test_a_receive_waiting_longer_than_the_timeout_raises_with_a_report() -> None:
    comm_channel = InProcessCommChannel(timeout=0.05)
    released = threading.Event()

    def a() -> None:
        try:
            comm_channel.receive("b", "a", "x")
        finally:
            released.set()

    # b computes (in no receive) until a gave up: no deadlock to tell
    with pytest.raises(CouplingTimeout) as raised:
        run_regions({"a": a, "b": released.wait}, comm_channel)

    report = str(raised.value)
    assert report.startswith(
        "a coupling timeout: region a waited 0.05 s for 'x' from region b (commChannelTimeout)"
    )
    assert _waits(report) == ["a waits for 'x' from b", "b is busy (in no receive)"]
    assert report.endswith("undelivered messages: none")


@pytest.mark.parametrize("timeout", [None, 0])
def test_a_comm_channel_without_timeout_waits_without_end(timeout: Any) -> None:
    assert InProcessCommChannel(timeout=timeout).timeout is None


def test_the_timeout_is_ten_minutes_by_default() -> None:
    assert InProcessCommChannel().timeout == 600.0


def test_a_negative_timeout_raises() -> None:
    with pytest.raises(ValueError, match="at least 0 seconds, not -1"):
        InProcessCommChannel(timeout=-1)


def test_the_tags_of_the_coupling_messages() -> None:
    assert Tag.data("x") == "data:x"
    assert Tag.side("x") == "side:x"
    assert Tag.node("outer", "decision") == "outer/decision"
    assert Tag.TIME_DT == "time/dt"


def test_a_receiver_gets_a_copy() -> None:
    comm_channel = InProcessCommChannel()
    sent = {"T": np.ones(3), "h": 2.0}

    comm_channel.send("a", "b", "x", sent)
    received = comm_channel.receive("a", "b", "x")
    received["T"][0] = 5.0

    assert received is not sent and sent["T"][0] == 1.0 and received["h"] == 2.0


def test_a_message_that_is_not_plain_data_raises() -> None:
    comm_channel = InProcessCommChannel()

    with pytest.raises(TypeError, match="the message 'x' from region a to region b is not plain"):
        comm_channel.send("a", "b", "x", SimpleNamespace(f=lambda: None))
