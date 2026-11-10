# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Each selection of a pressure-velocity algorithm is a run of its own.

Two regions or runs select the same spec; their settings live on the runtime each
``create`` returns, so the module-level spec stays unchanged.
"""

from __future__ import annotations

import pytest

from neofoam.framework.initialization import ConfigContext
from neofoam.solver.incompressibleFluid.models.boussinesq import resolve as boussinesq_resolve
from neofoam.solver.incompressibleFluid.models.pressure_velocity.base import (
    PressureVelocityAlgorithm,
)
from neofoam.solver.incompressibleFluid.models.pressure_velocity.pimpleAlgorithm import pimple
from neofoam.solver.incompressibleFluid.models.pressure_velocity.simpleAlgorithm import simple


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
