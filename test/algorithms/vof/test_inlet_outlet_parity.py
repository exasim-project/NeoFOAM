# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the scalar ``inletOutlet`` patch evaluated with the face flux.

OpenFOAM's inletOutlet looks up ``phi`` and sets the inlet value on faces with
``phi < 0`` and the zero-gradient value elsewhere; NeoN's ``InletOutlet`` does the
same only when ``phi`` reaches it through the BoundaryContext
(``correct_boundary_conditions_with_flux``, which the NeoN MULES ops call after every
alpha solve). The damBreak comparisons keep inletOutlet on ``alpha.water`` at the
atmosphere, so this is the patch type their result depends on.

The field ``alphaInletOutlet`` has inletOutlet (inlet value 0.25) on every patch that
carries faces; the seeded velocity is a solid-body rotation with a vertical drift, so
with the zeroGradient ``U`` the boundary flux takes both signs on every mesh, and the
seeded alpha (between 0.5 and 0.9) differs from the inlet value everywhere.

Tolerance: none. Each face value is either the inlet value or the owner cell's value,
picked by the sign of ``phi``, which is bit-identical on both backends (``nfb.create_phi``
runs OpenFOAM's ``fvc::flux``), so no arithmetic can round differently.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from ._parity_backends import _Simulation
from ._two_phase_backends import _flux, nb, pyb
from .conftest import EXECUTORS, MESH_NAMES

_INLET_VALUE = 0.25


def _seed(sim: _Simulation) -> None:
    """A varying alpha and a rotating velocity with drift, with its phi."""
    centres = sim.mesh.cell_centres
    offset = centres - centres.mean(axis=0)
    x, y, _ = offset.T
    sim.field("alphaInletOutlet")[:] = 0.7 + 0.2 * np.sin(3.0 * x) * np.cos(2.0 * y)
    sim.field("U")[:] = np.column_stack([-y, x, np.full_like(x, 0.2)])
    _flux(sim.field("U"))


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", [*MESH_NAMES, "damBreak"])
def test_inlet_outlet_with_flux_matches_openfoam(
    two_phase_simulation: Callable[[str, str], _Simulation], mesh: str, executor: str
) -> None:
    sim = two_phase_simulation(mesh, executor)
    _seed(sim)
    alpha = sim.field("alphaInletOutlet")

    expected = pyb.correct_with_flux(alpha)
    result = nb.correct_with_flux(alpha)

    # without both face kinds the parity below would not exercise the flux switch
    inflow = np.count_nonzero(expected == _INLET_VALUE)
    assert 0 < inflow < expected.size, f"{mesh}: {inflow} of {expected.size} faces are inflow"
    np.testing.assert_array_equal(result, expected, err_msg=f"{mesh}/{executor}")
