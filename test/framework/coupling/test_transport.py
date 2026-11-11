# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The in-process transport tells a deadlock at once and times out as a fallback.

Regions ``a``, ``b`` (and ``c``) are plain tasks run by :func:`run_regions`,
each receiving from or sending to the others on the transport directly.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from neofoam.framework.coupling import (
    CouplingDeadlock,
    CouplingTimeout,
    CouplingTransportError,
    InProcessTransport,
    TransportClosed,
)
from neofoam.framework.coupling.region import run_regions
from neofoam.framework.coupling.transport import Tag


def _waits(lines: str) -> list[str]:
    """The lines of a report that tell a region's state, without their durations."""
    return [line.split(" (for")[0].strip() for line in lines.splitlines() if line.startswith("  ")]


def _errors(transport: InProcessTransport, tasks: dict[str, Callable[[], Any]]) -> dict[str, Any]:
    """Run ``tasks`` on their own threads as run_regions does; the error each raised, by region."""
    errors: dict[str, Any] = {}

    def run(region: str, task: Callable[[], Any]) -> None:
        try:
            task()
        except CouplingTransportError as error:
            errors[region] = error
        finally:
            transport.close(region)

    threads = [threading.Thread(target=run, args=item) for item in tasks.items()]
    transport.start(tasks)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_two_regions_receiving_from_each_other_raise_a_deadlock_at_once() -> None:
    transport = InProcessTransport()
    started = time.monotonic()

    errors = _errors(
        transport,
        {
            "a": lambda: transport.receive("b", "a", "x"),
            "b": lambda: transport.receive("a", "b", "y"),
        },
    )

    assert time.monotonic() - started < 0.5
    assert set(errors) == {"a", "b"}
    assert all(isinstance(error, CouplingDeadlock) for error in errors.values())
    report = str(errors["a"])
    assert report.startswith("a coupling deadlock: every running region waits")
    assert _waits(report) == ["a waits for 'x' from b", "b waits for 'y' from a"]
    assert report.endswith("undelivered messages: none")


def test_a_deadlock_lists_the_messages_sent_but_not_received() -> None:
    transport = InProcessTransport()

    def a() -> None:
        transport.send("a", "b", "z", 1.0)
        transport.send("a", "b", "z", 2.0)
        transport.receive("b", "a", "x")

    errors = _errors(transport, {"a": a, "b": lambda: transport.receive("a", "b", "y")})

    assert str(errors["b"]).endswith("undelivered messages: a→b 'z' (2)")


def test_run_regions_raises_the_deadlock() -> None:
    transport = InProcessTransport()

    with pytest.raises(CouplingDeadlock, match="a waits for 'x' from b"):
        run_regions(
            {
                "a": lambda: transport.receive("b", "a", "x"),
                "b": lambda: transport.receive("a", "b", "y"),
            },
            transport,
        )


def test_run_regions_raises_the_error_before_a_transport_closed_of_another_region() -> None:
    def closed() -> None:
        raise TransportClosed("a follows")

    def deadlock() -> None:
        time.sleep(0.05)
        raise CouplingDeadlock("b deadlocked")

    with pytest.raises(CouplingDeadlock, match="b deadlocked"):
        run_regions({"a": closed, "b": deadlock}, InProcessTransport())


def test_waiting_for_a_region_that_finished_raises_a_deadlock_at_once() -> None:
    transport = InProcessTransport()
    transport.start(["a", "b"])
    transport.close("b")

    with pytest.raises(
        CouplingDeadlock, match="region a waits for 'x' from region b, which finished"
    ):
        transport.receive("b", "a", "x")


def test_a_message_a_finished_region_sent_is_still_received() -> None:
    transport = InProcessTransport()
    transport.start(["a", "b"])
    transport.send("b", "a", "x", 1.0)
    transport.close("b")

    assert transport.receive("b", "a", "x") == 1.0


def test_waiting_for_a_region_that_failed_raises_transport_closed() -> None:
    transport = InProcessTransport()
    transport.start(["a", "b"])
    transport.close("b")  # finished first: aborting leaves it finished
    transport.abort(["a", "b", "c"])

    with pytest.raises(TransportClosed, match="region a waits for 'y' from region c"):
        transport.receive("c", "a", "y")
    with pytest.raises(CouplingDeadlock, match="which finished"):
        transport.receive("b", "a", "x")


def test_a_region_that_finished_is_neither_running_nor_waiting() -> None:
    transport = InProcessTransport()

    def a() -> None:
        time.sleep(0.05)  # c finished by then
        transport.receive("b", "a", "x")

    errors = _errors(
        transport, {"a": a, "b": lambda: transport.receive("a", "b", "y"), "c": lambda: None}
    )

    assert _waits(str(errors["a"])) == ["a waits for 'x' from b", "b waits for 'y' from a"]


def test_a_region_waiting_while_its_partner_computes_waits() -> None:
    transport = InProcessTransport()
    received: list[float] = []

    def b() -> None:
        time.sleep(0.3)
        transport.send("b", "a", "x", 1.0)

    run_regions(
        {
            "a": lambda: received.append(transport.receive("b", "a", "x")),
            "b": b,
        },
        transport,
    )

    assert received == [1.0]


def test_a_receive_waiting_longer_than_the_timeout_raises_with_a_report() -> None:
    transport = InProcessTransport(timeout=0.2)
    started = time.monotonic()

    def b() -> None:
        time.sleep(0.4)  # computes: no deadlock to tell

    with pytest.raises(CouplingTimeout) as raised:
        run_regions({"a": lambda: transport.receive("b", "a", "x"), "b": b}, transport)

    assert time.monotonic() - started < 1.0
    report = str(raised.value)
    assert report.startswith(
        "a coupling timeout: region a waited 0.2 s for 'x' from region b (transportTimeout)"
    )
    assert _waits(report) == ["a waits for 'x' from b", "b is busy (in no receive)"]
    assert "(for 0.2 s)" in report
    assert report.endswith("undelivered messages: none")


@pytest.mark.parametrize("timeout", [None, 0])
def test_a_transport_without_timeout_waits_without_end(timeout: Any) -> None:
    assert InProcessTransport(timeout=timeout).timeout is None


def test_the_timeout_is_ten_minutes_by_default() -> None:
    assert InProcessTransport().timeout == 600.0


def test_a_negative_timeout_raises() -> None:
    with pytest.raises(ValueError, match="at least 0 seconds, not -1"):
        InProcessTransport(timeout=-1)


def test_the_tags_of_the_coupling_messages() -> None:
    assert Tag.data("x") == "data:x"
    assert Tag.side("x") == "side:x"
    assert Tag.node("outer", "decision") == "outer/decision"
    assert Tag.TIME_DT == "time/dt"


def test_a_receiver_gets_a_copy() -> None:
    transport = InProcessTransport()
    sent = {"T": np.ones(3), "h": 2.0}

    transport.send("a", "b", "x", sent)
    received = transport.receive("a", "b", "x")
    received["T"][0] = 5.0

    assert received is not sent and sent["T"][0] == 1.0 and received["h"] == 2.0


def test_a_message_that_is_not_plain_data_raises() -> None:
    transport = InProcessTransport()

    with pytest.raises(TypeError, match="the message 'x' from region a to region b is not plain"):
        transport.send("a", "b", "x", SimpleNamespace(f=lambda: None))
