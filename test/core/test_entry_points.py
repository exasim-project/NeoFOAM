# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``neofoam.plugins`` loading, with the installed entry points replaced by on-disk test modules.

Real installed distributions are not used: the result would depend on what else is
installed in the environment. The modules live under ``cases/plugin_modules``.
"""

from importlib import metadata
from pathlib import Path

import pytest

import neofoam.core.entry_points as entry_points
from neofoam.core.plugin_system import PluginSystem

PLUGIN_MODULES = Path(__file__).parent / "cases" / "plugin_modules"


def _install(monkeypatch: pytest.MonkeyPatch, *installed: metadata.EntryPoint) -> None:
    """Make ``installed`` the only entry points, with a fresh load state."""
    monkeypatch.syspath_prepend(str(PLUGIN_MODULES))
    monkeypatch.setattr(entry_points, "_loaded", set())
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda group: [ep for ep in installed if ep.group == group],
    )


def _plugin(name: str, module: str) -> metadata.EntryPoint:
    return metadata.EntryPoint(name=name, value=module, group=entry_points.PLUGIN_GROUP)


def test_load_plugins_registers_the_plugin_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _plugin("test", "entry_point_test_plugin"))

    entry_points.load_plugins()

    registry = PluginSystem.get_registered("entryPointTestFamily")
    assert registry is not None
    assert registry.get_plugin_names() == ["addOne"]


def test_load_plugins_imports_each_plugin_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _plugin("test", "entry_point_test_plugin"))

    first = entry_points.load_plugins()
    second = entry_points.load_plugins()

    assert (first, second) == (["test"], [])


def test_load_plugins_names_the_failing_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _plugin("broken", "entry_point_broken_plugin"))

    with pytest.raises(RuntimeError, match="neofoam plugin 'broken'"):
        entry_points.load_plugins()
