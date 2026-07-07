# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The process-global interface path is gone — only model-owned interfaces remain."""

import importlib
from pathlib import Path

import pytest

import neofoam
from neofoam.framework.context import Context


def test_framework_interface_package_is_removed() -> None:
    with pytest.raises(ImportError):
        importlib.import_module("neofoam.framework.interface")


def test_interface_step_is_no_longer_importable() -> None:
    import neofoam.framework.initialization as init_pkg

    assert not hasattr(init_pkg, "interface_step")


def test_context_has_no_interfaces_field() -> None:
    ctx = Context(fields={}, models={})
    assert "interfaces" not in type(ctx).model_fields


def test_source_tree_is_free_of_the_global_interface_symbols() -> None:
    root = Path(neofoam.__file__).parent
    needles = ("ctx.interfaces", "interface_step", "InterfaceSpec", "BoundInterface")
    offenders: dict[str, list[str]] = {}
    for path in root.rglob("*.py"):
        text = path.read_text()
        hits = [n for n in needles if n in text]
        if hits:
            offenders[str(path.relative_to(root))] = hits
    assert not offenders, offenders
