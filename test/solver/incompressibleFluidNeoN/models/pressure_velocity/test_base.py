# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Programmatic selection of a NeoN pressure-velocity algorithm.

``PressureVelocityAlgorithmNeoN.create`` is the case-free entry point;
``detect_and_create`` is the on-disk one and needs a case, so only ``create`` is
covered here. It hands back the spec module singleton and stamps
``algorithm_type`` on it, which the solver reads back later — so both are asserted.
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

    assert model is pimpleNeoN
    assert model.algorithm_type == "PIMPLE"


@pytest.mark.parametrize("algorithm_type", ["SIMPLE", "Simple"])
def test_create_selects_the_simple_spec(algorithm_type: str) -> None:
    """Either spelling of SIMPLE selects the steady-state spec."""
    model = PressureVelocityAlgorithmNeoN.create(algorithm_type=algorithm_type)

    assert model is simpleNeoN
    assert model.algorithm_type == "SIMPLE"


@pytest.mark.parametrize("algorithm_type", ["PISO", "SIMPLEC"])
def test_create_rejects_an_unsupported_algorithm(algorithm_type: str) -> None:
    """Only PIMPLE and SIMPLE are selectable by name, and the error says so.

    ``PISO`` is a *case* shape, not a spec: ``detect_and_create`` maps a pisoFoam
    case onto the PIMPLE spec, which then reads its PISO control block.
    """
    with pytest.raises(ValueError, match="only supports the PIMPLE and SIMPLE"):
        PressureVelocityAlgorithmNeoN.create(algorithm_type=algorithm_type)
