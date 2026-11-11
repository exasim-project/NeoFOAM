# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Programmatic selection of a NeoN pressure-velocity algorithm.

``PressureVelocityAlgorithmNeoN.create`` is the case-free entry point;
``detect_and_create`` is the on-disk one and needs a case, so only ``create`` is
covered here. Each call returns a fresh runtime whose ``config`` carries the
run's ``algorithm_type``; the module-level spec stays unchanged.
"""

from __future__ import annotations

import pytest

from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.base import (
    PressureVelocityAlgorithmNeoN,
)
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.pimpleAlgorithm import (
    pimpleNeoN,
)
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.simpleAlgorithm import (
    simpleNeoN,
)


@pytest.mark.parametrize("algorithm_type", ["PIMPLE", "Pimple"])
def test_create_selects_the_pimple_spec(algorithm_type: str) -> None:
    """Either spelling of PIMPLE selects the transient spec."""
    model = PressureVelocityAlgorithmNeoN.create(algorithm_type=algorithm_type)

    assert model.spec is pimpleNeoN
    assert model.config.algorithm_type == "PIMPLE"


@pytest.mark.parametrize("algorithm_type", ["SIMPLE", "Simple"])
def test_create_selects_the_simple_spec(algorithm_type: str) -> None:
    """Either spelling of SIMPLE selects the steady-state spec."""
    model = PressureVelocityAlgorithmNeoN.create(algorithm_type=algorithm_type)

    assert model.spec is simpleNeoN
    assert model.config.algorithm_type == "SIMPLE"


@pytest.mark.parametrize(
    ("algorithm_type", "spec"), [("SIMPLE", simpleNeoN), ("PIMPLE", pimpleNeoN)]
)
def test_create_keeps_the_run_state_off_the_spec(algorithm_type: str, spec: object) -> None:
    """Two selections of one spec are two runtimes; the spec gains no attribute."""
    first = PressureVelocityAlgorithmNeoN.create(algorithm_type=algorithm_type)
    second = PressureVelocityAlgorithmNeoN.create(algorithm_type=algorithm_type)

    assert first is not second
    assert first.spec is second.spec is spec
    assert not hasattr(spec, "algorithm_type")


@pytest.mark.parametrize("algorithm_type", ["PISO", "SIMPLEC"])
def test_create_rejects_an_unsupported_algorithm(algorithm_type: str) -> None:
    """Only PIMPLE and SIMPLE are selectable by name, and the error says so.

    ``PISO`` is a *case* shape, not a spec: ``detect_and_create`` maps a pisoFoam
    case onto the PIMPLE spec, which then reads its PISO control block.
    """
    with pytest.raises(ValueError, match="only supports the PIMPLE and SIMPLE"):
        PressureVelocityAlgorithmNeoN.create(algorithm_type=algorithm_type)
