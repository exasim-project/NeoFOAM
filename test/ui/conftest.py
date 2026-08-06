# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared fixtures for the wizard tests."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

import pytest

#: Nominal heartbeat period — the 20 ms tick the freeze was measured with.
_PERIOD = 0.02


@pytest.fixture
def heartbeat_ticks() -> Callable[[Callable[[], Any]], int]:
    """Count 20 ms heartbeat ticks while a wizard handler runs on the event loop.

    trame is single-threaded: a handler that does its blocking work on the loop
    stops the heartbeat dead (0 ticks) and the whole UI freezes with it, while one
    that hands the work to a worker thread lets it keep ticking. Use it on a
    handler whose slow part has been stubbed to a known duration::

        assert heartbeat_ticks(server.controller.load_geometry) >= 5
    """

    def measure(handler: Callable[[], Any]) -> int:
        async def drive() -> int:
            ticks = 0

            async def beat() -> None:
                nonlocal ticks
                while True:
                    await asyncio.sleep(_PERIOD)
                    ticks += 1

            pulse = asyncio.create_task(beat())
            await asyncio.sleep(3 * _PERIOD)  # let the heartbeat settle
            ticks = 0
            # A synchronous handler returns None — then nothing was awaited and
            # the tick count is what the frozen loop managed, i.e. zero.
            result = handler()
            if inspect.isawaitable(result):
                await result
            pulse.cancel()
            return ticks

        return asyncio.run(drive())

    return measure
