# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the ViscosityModel Protocol and the Newtonian stub."""

from typing import Any

from neofoam.viscosity.base import ViscosityModel
from neofoam.viscosity.models.newtonian import NewtonianModel


def test_newtonian_model_satisfies_protocol() -> None:
    assert isinstance(NewtonianModel(), ViscosityModel)


def test_viscosity_protocol_requires_full_surface() -> None:
    class Incomplete:
        def nu(self) -> Any:
            return 0.0

        # missing correct

    assert not isinstance(Incomplete(), ViscosityModel)
