# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared fixtures for the tool-registry tests.

The tool registry (:mod:`neofoam.tools.registry`) is process-global mutable
state populated by import-time self-registration. Tests that ``register_tool``
a stub must not leak it into ``available_tools()`` for the rest of the session,
so snapshot the registry contents before each test and restore them after.
"""

from typing import Iterator

import pytest

from neofoam.tools import registry


@pytest.fixture(autouse=True)
def restore_tool_registry() -> Iterator[None]:
    """Snapshot ``_REGISTRY`` and restore its contents around every test."""
    snapshot = dict(registry._REGISTRY)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(snapshot)
