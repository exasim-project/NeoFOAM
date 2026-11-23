# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The boussinesq resolve changes the run it resolves with, not the shared spec."""

from __future__ import annotations

from neofoam.framework.initialization import ConfigContext
from neofoam.solver.incompressibleFluid.models.boussinesq import resolve as boussinesq_resolve
from neofoam.solver.incompressibleFluid.models.pressure_velocity.base import (
    PressureVelocityAlgorithm,
)
from neofoam.solver.incompressibleFluid.models.pressure_velocity.pimpleAlgorithm import pimple


def test_boussinesq_flips_only_the_run_it_resolves_with() -> None:
    """The boussinesq resolve of one run leaves a second run of the same spec plain."""
    buoyant = PressureVelocityAlgorithm.create(algorithm_type="PIMPLE")
    plain = PressureVelocityAlgorithm.create(algorithm_type="PIMPLE")
    ctx = ConfigContext()
    ctx.register(buoyant.name, buoyant)

    boussinesq_resolve(None, ctx)

    assert buoyant is not plain
    assert buoyant.config.use_boussinesq is True
    assert plain.config.use_boussinesq is False
    assert not hasattr(pimple, "use_boussinesq")
