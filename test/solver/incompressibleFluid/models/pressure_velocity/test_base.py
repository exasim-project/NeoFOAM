# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A pressure-velocity selection is a runtime of its own; the spec stays unchanged."""

from __future__ import annotations

import pytest

from neofoam.solver.incompressibleFluid.models.pressure_velocity.base import (
    PressureVelocityAlgorithm,
)
from neofoam.solver.incompressibleFluid.models.pressure_velocity.pimpleAlgorithm import pimple
from neofoam.solver.incompressibleFluid.models.pressure_velocity.simpleAlgorithm import simple


@pytest.mark.parametrize(
    ("algorithm_type", "spec", "expected"),
    [("SIMPLE", simple, "SIMPLE"), ("Pimple", pimple, "PIMPLE")],
)
def test_create_keeps_the_algorithm_type_off_the_spec(
    algorithm_type: str, spec: object, expected: str
) -> None:
    """The selected algorithm is the runtime's setting; the spec gains no attribute."""
    runtime = PressureVelocityAlgorithm.create(algorithm_type=algorithm_type)

    assert runtime.spec is spec
    assert runtime.config.algorithm_type == expected
    assert not hasattr(spec, "algorithm_type")
