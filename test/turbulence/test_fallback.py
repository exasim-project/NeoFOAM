# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the OpenFOAM turbulence fallback adapter.

These run without pybFoam: the factory is injected, and the lazy pybFoam import
only fires when ``build()`` is called without a factory.
"""

import builtins
from typing import Any
from unittest.mock import MagicMock

import pytest

from neofoam.turbulence.fallback import OpenFOAMTurbulenceModel
from neofoam.turbulence.stress import OpenFOAMStress


def test_fallback_calls_injected_factory_with_fields() -> None:
    factory = MagicMock(return_value=MagicMock())
    model = OpenFOAMTurbulenceModel("U", "phi", "transport", factory=factory)

    model.build()

    factory.assert_called_once_with("U", "phi", "transport")


def test_fallback_delegates_to_impl() -> None:
    impl = MagicMock()
    impl.nut.return_value = "nut"
    impl.nu.return_value = "nu"
    model = OpenFOAMTurbulenceModel("U", "phi", "transport", factory=lambda *a: impl)

    model.build()

    # nut/nu/correct delegate to the wrapped pybFoam model; divDevReff is the
    # shared linear viscous stress (covered end-to-end by the comparison test).
    assert model.nut() == "nut"
    assert model.nu() == "nu"
    model.correct()
    impl.correct.assert_called_once_with()


def test_fallback_defines_openfoam_stress() -> None:
    # The fallback is a peer model that defines its own stress, delegating
    # divDevReff to the wrapped pybFoam model.
    impl = MagicMock()
    impl.divDevReff.return_value = "div"
    model = OpenFOAMTurbulenceModel("U", "phi", "transport", factory=lambda *a: impl)
    model.build()

    stress = model.viscous_stress()
    assert isinstance(stress, OpenFOAMStress)
    assert stress.divDevReff("U") == "div"
    impl.divDevReff.assert_called_once_with("U")


def test_fallback_corrects_after_the_loop() -> None:
    # The pybFoam model owns its eddy viscosity; it contributes one after-loop
    # ``correct`` operation that advances the model.
    model = OpenFOAMTurbulenceModel(
        "U", "phi", "transport", factory=lambda *a: object()
    )

    ops = model.operations
    assert len(ops) == 1
    assert ops[0].operation_name == "of_correct_turbulence"


def test_use_before_build_raises() -> None:
    model = OpenFOAMTurbulenceModel(
        "U", "phi", "transport", factory=lambda *a: object()
    )
    with pytest.raises(RuntimeError):
        model.nut()


def test_construction_does_not_import_pybfoam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the adapter must not import pybFoam; only a factory-less
    build() does (and then surfaces the missing dependency)."""
    real_import = builtins.__import__

    def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pybFoam" or name.startswith("pybFoam."):
            raise ImportError("pybFoam blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    # Construction is side-effect free even with pybFoam unavailable.
    model = OpenFOAMTurbulenceModel("U", "phi", "transport")

    # A factory-less build() reaches the lazy import and fails cleanly.
    with pytest.raises(ImportError):
        model.build()
