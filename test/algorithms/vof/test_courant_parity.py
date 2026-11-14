# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the VoF Courant numbers (pybFoam vs NeoFOAM bindings).

``nfb.courant_numbers`` must return what CourantNo.H and advectorVoF's
alphaCourantNo.H give: ``0.5*gMax(surfaceSum(|phi|)/V)*deltaT``, the second
over the cells with ``0.01 < alpha < 0.99``. The flux is a solid-body
rotation with a vertical drift, so the face fluxes take every sign and size
and the zeroGradient walls carry flux too (the boundary faces enter the sum).
Two phase fractions: a sharp sphere (alpha 0 or 1 only, so no interface cell
and alphaCoNum is 0) and a smooth tanh sphere, whose interface band sits away
from the fastest cells, so the two Courant numbers differ.

The smooth sphere is the one input still seeded in Python: setExprFields'
``tanh`` differs from numpy's in the last bits (up to 1e-15), and this test
compares doubles exactly, so its input stays the one it always was.

Tolerance: none. The binding sums every cell's faces in OpenFOAM's order
(a per-cell gather, no atomics) and takes the maximum, which is
order-independent, so the result is OpenFOAM's double on every executor.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from ._parity_backends import _Field, _Simulation, _use_velocity, nb, pyb
from .conftest import EXECUTORS, MESH_NAMES

DELTA_T = 0.02
SPHERE_RADIUS = 0.3
INTERFACE_WIDTH = 0.05


def _phase_fraction(sim: _Simulation, interface_width: float) -> _Field:
    """The case's sharp sphere, or the tanh sphere of this width at the mesh centroid."""
    if interface_width == 0:
        return sim.field("alpha")
    centres = sim.mesh.cell_centres
    distance = np.linalg.norm(centres - centres.mean(axis=0), axis=1) - SPHERE_RADIUS
    smooth = sim.field("alphaSmooth")
    smooth[:] = 0.5 * (1.0 - np.tanh(distance / interface_width))
    return smooth


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("interface_width", [0.0, INTERFACE_WIDTH], ids=["sharp", "smooth"])
def test_courant_numbers_match_openfoam(
    parity_simulation: Callable[[str, str], _Simulation],
    mesh: str,
    executor: str,
    interface_width: float,
) -> None:
    sim = parity_simulation(mesh, executor)
    _use_velocity(sim)
    alpha = _phase_fraction(sim, interface_width)

    expected = pyb.vof.courant_numbers(alpha, DELTA_T)
    result = nb.vof.courant_numbers(alpha, DELTA_T)

    # the seed itself: interface cells exactly for the smooth sphere
    assert (expected[1] > 0) == (interface_width > 0), f"{mesh}: alphaCoNum {expected[1]}"
    np.testing.assert_array_equal(
        result, expected, err_msg=f"{mesh}/{executor} interface width {interface_width}"
    )
