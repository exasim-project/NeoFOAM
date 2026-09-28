# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the OpenFOAM viscosity (transport) fallback adapter.

These run without pybFoam: the factory is injected, and the lazy pybFoam import
only fires when ``build()`` is called without a factory.
"""

import builtins
from typing import Any
from unittest.mock import MagicMock

import pytest

from neofoam.viscosity.fallback import OpenFOAMViscosityModel


def test_fallback_calls_injected_factory_with_fields() -> None:
    factory = MagicMock(return_value=MagicMock())
    model = OpenFOAMViscosityModel("U", "phi", factory=factory)

    model.build()

    factory.assert_called_once_with("U", "phi")


def test_fallback_delegates_to_impl() -> None:
    impl = MagicMock()
    impl.nu.return_value = "nu"
    model = OpenFOAMViscosityModel("U", "phi", factory=lambda *a: impl)

    model.build()

    assert model.nu() == "nu"
    model.correct()
    impl.correct.assert_called_once_with()


def test_use_before_build_raises() -> None:
    model = OpenFOAMViscosityModel("U", "phi", factory=lambda *a: object())
    with pytest.raises(RuntimeError):
        model.nu()


def test_construction_does_not_import_pybfoam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the adapter must not import pybFoam; only a factory-less
    build() does (and then surfaces the missing dependency)."""
    real_import = builtins.__import__

    def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pybFoam" or name.startswith("pybFoam."):
            raise ImportError("pybFoam blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    model = OpenFOAMViscosityModel("U", "phi")

    with pytest.raises(ImportError):
        model.build()
