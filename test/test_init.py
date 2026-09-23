# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``import neofoam`` with the compiled bindings absent or broken.

An import finder stands in for the two states of ``neofoam.neofoam_bindings`` — a
broken shared object cannot be built on demand — and ``importlib.reload`` re-runs
``neofoam/__init__.py`` against it. ``monkeypatch`` restores ``sys.modules``,
``sys.meta_path`` and the package attribute, so the real bindings are back afterwards.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

import neofoam

_BINDINGS = "neofoam.neofoam_bindings"


class _Finder:
    """Raises ``error`` for the bindings; every other import is left alone."""

    def __init__(self, error: ImportError) -> None:
        self._error = error

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == _BINDINGS:
            raise self._error


def _reimport_with(monkeypatch: pytest.MonkeyPatch, error: ImportError) -> None:
    monkeypatch.delattr(neofoam, "neofoam_bindings")
    monkeypatch.delitem(sys.modules, _BINDINGS)
    monkeypatch.setattr(sys, "meta_path", [_Finder(error), *sys.meta_path])


def test_import_without_built_bindings_leaves_them_none(monkeypatch):
    _reimport_with(
        monkeypatch, ModuleNotFoundError(f"No module named '{_BINDINGS}'", name=_BINDINGS)
    )

    importlib.reload(neofoam)

    assert neofoam.neofoam_bindings is None


def test_import_with_broken_bindings_raises_the_real_cause(monkeypatch):
    _reimport_with(monkeypatch, ImportError("undefined symbol: nb_type_lookup"))

    with pytest.raises(ImportError, match="undefined symbol: nb_type_lookup"):
        importlib.reload(neofoam)
