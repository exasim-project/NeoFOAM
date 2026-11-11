# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""run_regions runs every region on a thread of its own and raises the first real error."""

from __future__ import annotations

import pytest
from framework.coupling.regions import TIMEOUT

from neofoam.framework.coupling import (
    CommChannelClosed,
    CouplingDeadlock,
    InProcessCommChannel,
)
from neofoam.framework.coupling.run import run_regions


def test_run_regions_raises_the_deadlock() -> None:
    comm_channel = InProcessCommChannel()

    with pytest.raises(CouplingDeadlock, match="a waits for 'x' from b"):
        run_regions(
            {
                "a": lambda: comm_channel.receive("b", "a", "x"),
                "b": lambda: comm_channel.receive("a", "b", "y"),
            },
            comm_channel,
        )


def test_run_regions_raises_the_error_before_a_comm_channel_closed_of_another_region() -> None:
    comm_channel = InProcessCommChannel(timeout=TIMEOUT)

    def closed() -> None:
        raise CommChannelClosed("a follows")

    def deadlock() -> None:
        # returns only once a failed: b's error comes second
        with pytest.raises(CommChannelClosed):
            comm_channel.receive("a", "b", "x")
        raise CouplingDeadlock("b deadlocked")

    with pytest.raises(CouplingDeadlock, match="b deadlocked"):
        run_regions({"a": closed, "b": deadlock}, comm_channel)
