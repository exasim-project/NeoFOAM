# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Fake coupling plugins: entry points of the group ``neofoam.coupling``, no installed package.

:func:`installed` makes ``importlib.metadata.entry_points`` list the given plugins
and lets :func:`~neofoam.framework.coupling.load_plugins` import them anew; on
exit the strategy families hold the built-in strategies again. A plugin is the
source of its module (written to ``directory``, on ``sys.path``, as an
installed package's would be) or, without a line break, an entry point's value.

Example::

    with installed(monkeypatch, tmp_path, precice=SOURCE) as asked:
        load_plugins()
    assert asked == [ENTRY_POINT_GROUP]   # entry_points read once
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.coupling import kernels
from neofoam.framework.coupling.kernels import ENTRY_POINT_GROUP

#: The strategy families a plugin may register with.
FAMILIES = ("CouplingMapping", "CouplingAcceleration", "CouplingConvergence")


@contextmanager
def installed(
    monkeypatch: pytest.MonkeyPatch, directory: Path, **plugins: str
) -> Iterator[list[str]]:
    """The plugins ``plugins`` by entry-point name; yields the groups ``entry_points`` was asked."""
    found = []
    modules = []
    for name, plugin in plugins.items():
        value = plugin
        if "\n" in plugin:
            value = f"fake_coupling_plugin_{name}"
            (directory / f"{value}.py").write_text(plugin)
            modules.append(value)
        found.append(importlib.metadata.EntryPoint(name, value, ENTRY_POINT_GROUP))
    monkeypatch.syspath_prepend(str(directory))
    importlib.invalidate_caches()
    asked: list[str] = []

    def entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
        asked.append(group)
        return [entry_point for entry_point in found if entry_point.group == group]

    built_in = {family: list(PluginSystem.list_plugins()[family]) for family in FAMILIES}
    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    monkeypatch.setattr(kernels, "_loaded", False)
    try:
        yield asked
    finally:
        for family, members in built_in.items():
            for member in list(PluginSystem.list_plugins()[family]):
                if member not in members:
                    PluginSystem.remove_plugin_model(family, member)
        for module in modules:
            sys.modules.pop(module, None)
