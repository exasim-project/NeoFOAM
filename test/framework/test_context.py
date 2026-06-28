# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from neofoam.framework.context import Context


def test_context_defaults() -> None:
    ctx = Context(fields={}, models={})

    assert ctx.fields == {}
    assert ctx.models == {}
    assert ctx.mesh is None
    assert ctx.time is None


def test_context_time_attribute() -> None:
    sentinel = object()
    ctx = Context(fields={}, models={}, time=sentinel)

    assert ctx.time is sentinel


def test_mesh_stats_defaults_none() -> None:
    ctx = Context(fields={}, models={})
    assert ctx.mesh_stats is None
