# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared fixtures for the wizard tests."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

import pytest

from neofoam.mcp import tools
from neofoam.mcp.registry import resolve_solver
from neofoam.ui import build_app

#: Nominal heartbeat period — the 20 ms tick the freeze was measured with.
_PERIOD = 0.02


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect the ``browser`` tests unless the ``-m`` expression names the marker.

    Not ``addopts = -m "not browser"``: a command-line ``-m "not slow"`` replaces that
    expression, and the default run must never start a browser.
    """
    if "browser" in config.getoption("markexpr"):
        return
    browser = [item for item in items if item.get_closest_marker("browser")]
    if browser:
        items[:] = [item for item in items if item not in browser]
        config.hook.pytest_deselected(items=browser)


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


@pytest.fixture
def solver() -> Any:
    """The ``incompressibleFluid`` solver most wizard tests build their forms from."""
    return resolve_solver("incompressibleFluid")


@pytest.fixture
def wizard(request: pytest.FixtureRequest) -> Any:
    """A default wizard server without step plugins, named after the requesting test.

    trame keeps one server (and its state) per name for the whole process, so a
    copy-pasted name silently shares state between two tests; the node name is
    unique by construction, parametrized ids included.
    """
    get_server = pytest.importorskip("trame.app").get_server
    return build_app(server=get_server(request.node.name), plugins=[])


@pytest.fixture
def seed_transport_defaults(solver: Any) -> Callable[[Any], None]:
    """Fill a wizard server with the minimal valid form state (only transportProperties)."""
    defaults = tools.config_schema(solver, "transport_properties_config").defaults

    def seed(server: Any) -> None:
        for entry in server.controller.get_entries():
            server.state[entry.state_key] = (
                {**defaults, "nu": 1e-05}
                if entry.config_name == "transport_properties_config"
                else {}
            )

    return seed
