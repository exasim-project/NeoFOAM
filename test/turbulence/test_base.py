# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the TurbulenceModel Protocol and the laminar stub."""

from typing import Any

from neofoam.turbulence.base import TurbulenceModel
from neofoam.turbulence.models.laminar import LaminarModel


def test_laminar_model_satisfies_protocol() -> None:
    assert isinstance(LaminarModel(), TurbulenceModel)


def test_turbulence_protocol_requires_full_surface() -> None:
    class Incomplete:
        def nut(self) -> Any:
            return 0.0

        def nu(self) -> Any:
            return 0.0

        # missing divDevReff and correct

    assert not isinstance(Incomplete(), TurbulenceModel)
