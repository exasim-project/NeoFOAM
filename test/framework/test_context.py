# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from neofoam.framework.context import Context


def test_context_defaults():
    ctx = Context(fields={}, models={})

    assert ctx.fields == {}
    assert ctx.models == {}
    assert ctx.mesh is None
    assert ctx.runtime is None


def test_context_runtime_attribute():
    sentinel = object()
    ctx = Context(fields={}, models={}, runtime=sentinel)

    assert ctx.runtime is sentinel
