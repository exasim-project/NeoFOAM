# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``neofoam.plugins`` loading, with the installed entry points replaced by on-disk test modules.

Real installed distributions are not used: the result would depend on what else is
installed in the environment. The modules live under ``cases/plugin_modules``.
"""

import importlib
import sys
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path

import pytest

import neofoam.core.entry_points as entry_points
from neofoam.core.plugin_system import PluginSystem

PLUGIN_MODULES = Path(__file__).parent / "cases" / "plugin_modules"


@pytest.fixture(autouse=True)
def restore_plugin_state() -> Iterator[None]:
    """Forget the imported test plugins and what they registered, so each test starts clean."""
    before = {family: list(plugins) for family, plugins in PluginSystem.list_plugins().items()}
    yield
    for module in PLUGIN_MODULES.glob("*.py"):
        sys.modules.pop(module.stem, None)
    # Dropped whole: remove_plugin_model cannot rebuild a family left with no plugins.
    for family in set(PluginSystem.list_plugins()) - set(before):
        del PluginSystem._registry[family]
    for family, plugins in PluginSystem.list_plugins().items():
        for plugin_cls in list(plugins):
            if plugin_cls not in before[family]:
                PluginSystem.remove_plugin_model(family, plugin_cls)


def _install(monkeypatch: pytest.MonkeyPatch, *installed: metadata.EntryPoint) -> None:
    """Make ``installed`` the only entry points."""
    monkeypatch.syspath_prepend(str(PLUGIN_MODULES))
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda group: [entry_point for entry_point in installed if entry_point.group == group],
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
    _install(monkeypatch, _plugin("counting", "entry_point_counting_plugin"))

    entry_points.load_plugins()
    entry_points.load_plugins()

    assert importlib.import_module("entry_point_counting_plugin").import_count == 1


def test_load_plugins_imports_both_plugins_of_the_same_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        _plugin("shared", "entry_point_test_plugin"),
        _plugin("shared", "entry_point_counting_plugin"),
    )

    entry_points.load_plugins()

    assert {"entry_point_test_plugin", "entry_point_counting_plugin"} <= set(sys.modules)


def test_a_failed_plugin_fails_again_on_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _plugin("broken", "entry_point_broken_plugin"))

    for _ in range(2):
        with pytest.raises(RuntimeError, match="neofoam plugin 'broken'"):
            entry_points.load_plugins()
