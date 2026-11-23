# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Physical invariants of the NeoFOAM MULES explicit solve: boundedness and mass conservation.

The ``sharpSphere`` case's sphere is advected by the uniform translation. MULES
bounds alpha only for a divergence-free flux; with zeroGradient walls only the
uniform field gives one (the rotation's wall fluxes do not close), and a
fixedValue patch at an outflow removes alpha the cell does not hold, so these
tests use the zeroGradient ``alpha``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from ._parity_backends import _Simulation, _use_velocity, nb
from .conftest import EXECUTORS, MESH_NAMES

DELTA_T = 0.02  # Courant number below 0.3 on all meshes for |U| <= 0.75


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_mules_explicit_solve_stays_bounded(
    parity_simulation: Callable[[str, str], _Simulation], mesh: str, executor: str
) -> None:
    sim = parity_simulation(mesh, executor)
    _use_velocity(sim, rotate=False)

    result = nb.mules.explicit_solve(sim.field("alpha"), DELTA_T, 1.0, 20)

    # No clamping anywhere. MULES with the default nLimiterIter = 3 does not fully
    # converge its limiter: OpenFOAM itself undershoots by ~4e-8 here (the
    # "translation" parity case reproduces it), far below the jump of 1.
    assert result.min() >= -1e-6, f"{mesh}/{executor}: min {result.min()}"
    assert result.max() <= 1.0 + 1e-6, f"{mesh}/{executor}: max {result.max()}"


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_mules_explicit_solve_conserves_mass(
    parity_simulation: Callable[[str, str], _Simulation], mesh: str, executor: str
) -> None:
    """One step: the wall cells hold alpha = 0, so no alpha crosses the walls."""
    sim = parity_simulation(mesh, executor)
    _use_velocity(sim, rotate=False)
    alpha = sim.field("alpha")
    volumes = sim.mesh.cell_volumes
    mass = np.sum(volumes * np.asarray(alpha))

    result = nb.mules.explicit_solve(alpha, DELTA_T, 1.0, 1)

    np.testing.assert_allclose(
        np.sum(volumes * result), mass, rtol=1e-13, err_msg=f"{mesh}/{executor}"
    )
