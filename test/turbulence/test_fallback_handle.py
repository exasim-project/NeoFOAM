# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The pybFoam-fallback momentum-transport handle (:class:`FallbackHandle`).

The handle pairs an :class:`OpenFOAMTurbulenceModel` (nut/stress provider) with
the model file's ``fallback=True`` operations. These tests stub the OF model so
no OpenFOAM build is needed — they pin the forwarding surface and that the
handle yields the fallback ops as ``.operations``.
"""

from typing import Any

from neofoam.turbulence.fallback import FallbackHandle
from neofoam.turbulence.protocol import MomentumTransport


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


def test_forwards_nut_and_has_nut() -> None:
    of = _StubOF()
    handle = FallbackHandle(of, operations=[])
    assert handle.has_nut() is True
    assert handle.nut() == "nut-field"
    assert handle.nu() == "nu-field"


def test_operations_returns_the_fallback_ops() -> None:
    sentinel = object()
    handle = FallbackHandle(_StubOF(), operations=[sentinel])
    assert handle.operations == [sentinel]


def test_correct_and_build_delegate_to_of_model() -> None:
    of = _StubOF()
    handle = FallbackHandle(of, operations=[])
    handle.build()
    handle.correct()
    assert of.built == 1
    assert of.corrected == 1


def test_viscous_stress_and_divdevreff_delegate() -> None:
    handle = FallbackHandle(_StubOF(), operations=[])
    assert handle.viscous_stress() == "of-stress"
    assert handle.divDevReff("U") == "divDevReff(U)"


def test_satisfies_momentum_transport_protocol() -> None:
    handle = FallbackHandle(_StubOF(), operations=[])
    assert isinstance(handle, MomentumTransport)
