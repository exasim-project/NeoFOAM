# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Fake coupling plugins: entry points of the group ``neofoam.coupling``, no installed package.

:func:`installed` makes ``importlib.metadata.entry_points`` list the given plugins
and the registries load them anew on their next read; on exit the registries
hold the built-in kernels again. A plugin is its ``register`` callable (put in a
module of its own, as an installed package's would be) or an entry point's
``module:attribute`` text.

Example::

    with installed(monkeypatch, precice=register) as loaded:
        mapping("rbf")
    assert loaded == [ENTRY_POINT_GROUP]   # entry_points read once
"""

from __future__ import annotations

import importlib.metadata
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable, Union

import pytest

from neofoam.framework.coupling import (
    ACCELERATIONS,
    CONVERGENCES,
    ENTRY_POINT_GROUP,
    MAPPINGS,
    Registry,
    kernels,
)

Register = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None]


@contextmanager
def installed(
    monkeypatch: pytest.MonkeyPatch, **plugins: Union[Register, str]
) -> Iterator[list[str]]:
    """The plugins ``plugins`` by entry-point name; yields the groups ``entry_points`` was asked."""
    found = []
    for name, plugin in plugins.items():
        value = plugin
        if not isinstance(plugin, str):
            module = types.ModuleType(f"fake_coupling_plugin_{name}")
            module.register = plugin  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, module.__name__, module)
            value = f"{module.__name__}:register"
        found.append(importlib.metadata.EntryPoint(name, str(value), ENTRY_POINT_GROUP))
    asked: list[str] = []

    def entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
        asked.append(group)
        return [entry_point for entry_point in found if entry_point.group == group]

    registries: list[Registry[Any]] = [MAPPINGS, ACCELERATIONS, CONVERGENCES]
    built_in: list[dict[str, Any]] = [dict.copy(registry) for registry in registries]
    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    monkeypatch.setattr(kernels, "_plugins_loaded", None)
    try:
        yield asked
    finally:
        for registry, entries in zip(registries, built_in):
            dict.clear(registry)
            dict.update(registry, entries)
