# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the pybFoam-fallback path: the adapter and the handle that pairs it.

:class:`OpenFOAMTurbulenceModel` is the adapter that wraps a pybFoam turbulence
model; :class:`FallbackHandle` pairs one with the model file's ``fallback=True``
operations and is what ``incompressibleFluid`` holds.

Neither half needs pybFoam here: the adapter's factory is injected (the lazy pybFoam
import only fires when ``build()`` is called without one) and the handle is given a
stub adapter. What is pinned is the forwarding surface — the values themselves are
proven against real OpenFOAM by ``test_neon_turbulence_parity``.
"""

import builtins
from typing import Any
from unittest.mock import MagicMock

import pytest

from neofoam.turbulence.fallback import FallbackHandle, OpenFOAMTurbulenceModel
from neofoam.turbulence.protocol import MomentumTransport
from neofoam.turbulence.stress import OpenFOAMStress

# --------------------------------------------------------------------------- #
# The adapter                                                                  #
# --------------------------------------------------------------------------- #


def test_fallback_builds_through_the_factory_and_delegates() -> None:
    impl = MagicMock()
    impl.nut.return_value = "nut"
    impl.nu.return_value = "nu"
    factory = MagicMock(return_value=impl)
    model = OpenFOAMTurbulenceModel("U", "phi", "transport", factory=factory)

    model.build()

    factory.assert_called_once_with("U", "phi", "transport")
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
    model = OpenFOAMTurbulenceModel("U", "phi", "transport", factory=lambda *a: object())

    ops = model.operations
    assert len(ops) == 1
    assert ops[0].operation_name == "of_correct_turbulence"


def test_use_before_build_raises() -> None:
    model = OpenFOAMTurbulenceModel("U", "phi", "transport", factory=lambda *a: object())
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


# --------------------------------------------------------------------------- #
# The handle                                                                   #
# --------------------------------------------------------------------------- #


class _StubOF:
    """Minimal stand-in for OpenFOAMTurbulenceModel (no pybFoam)."""

    def __init__(self) -> None:
        self.corrected = 0
        self.built = 0

    def build(self) -> "_StubOF":
        self.built += 1
        return self

    def has_nut(self) -> bool:
        return True

    def nut(self) -> str:
        return "nut-field"

    def nu(self) -> str:
        return "nu-field"

    def viscous_stress(self) -> str:
        return "of-stress"

    def divDevReff(self, U: Any, nu: Any = None, nut: Any = None) -> str:
        return f"divDevReff({U})"

    def correct(self) -> None:
        self.corrected += 1


def test_handle_forwards_everything_to_the_of_model() -> None:
    of = _StubOF()
    handle = FallbackHandle(of, operations=[])

    assert handle.has_nut() is True
    assert handle.nut() == "nut-field"
    assert handle.nu() == "nu-field"
    assert handle.viscous_stress() == "of-stress"
    assert handle.divDevReff("U") == "divDevReff(U)"

    handle.build()
    handle.correct()
    assert of.built == 1
    assert of.corrected == 1


def test_handle_operations_returns_the_fallback_ops() -> None:
    sentinel = object()
    handle = FallbackHandle(_StubOF(), operations=[sentinel])
    assert handle.operations == [sentinel]


def test_handle_satisfies_momentum_transport_protocol() -> None:
    handle = FallbackHandle(_StubOF(), operations=[])
    assert isinstance(handle, MomentumTransport)
