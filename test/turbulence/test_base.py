# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the ``ViscousStress`` Protocol — the surface momentum consumes.

The momentum equation calls ``viscousStress.divDevReff(U)``; the native linear
assembly and the OpenFOAM-fallback delegate both satisfy this Protocol.
"""

from typing import Any

from neofoam.turbulence import LinearViscousStress
from neofoam.turbulence.base import ViscousStress


def test_linear_viscous_stress_satisfies_protocol() -> None:
    stress: Any = LinearViscousStress()
    assert isinstance(stress, ViscousStress)


def test_incomplete_class_is_not_a_viscous_stress() -> None:
    class Incomplete:
        def update(self, ctx: Any) -> None:
            return None

        # missing divDevReff

    assert not isinstance(Incomplete(), ViscousStress)
