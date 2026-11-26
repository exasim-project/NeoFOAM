# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The explicit viscous stress div(nu*dev2(T(grad(U)))) does not depend on the order of the patches.

``box2D_emptyFirst`` and ``box2D_emptyLast`` are the same 2D mesh (same points,
cells and internal faces); only the order of the patches ``frontAndBack``
(empty) and ``walls`` in ``constant/polyMesh/boundary`` differs. NeoN's
boundary faces leave out the empty patch, OpenFOAM's face list keeps it, so a
boundary face that is looked up in the full face list by its NeoN index gets
the area vector of another face once an empty patch precedes it (snappy +
splitMeshRegions/extrudeMesh meshes put front/back before the walls). Then the
walls' stress flux is wrong and a z component appears in a 2D flow.

The velocity is planar and divergent (U_z = 0), so the stress has x/y parts up
to ~2 and no z part. Both meshes run the same arithmetic on the same cells, so
the two results agree to round-off: ``ATOL`` is 1e-12 of the stress peak.
"""

from __future__ import annotations

import numpy as np
import pytest
from backends import nb
from case_setup import simulation
from conftest import EXECUTORS

MESHES = ("box2D_emptyLast", "box2D_emptyFirst")
ATOL = 1e-12


def _stress(mesh: str, executor: str) -> np.ndarray:
    """The viscous stress of the planar, divergent velocity (x, y**2, 0) on ``mesh``."""
    sim = simulation(mesh, executor)
    x, y, _ = sim.mesh.cell_centres.T
    U = sim.field("U")
    U[:] = np.column_stack([x, y**2, np.zeros_like(x)])
    return nb.exp.viscous_stress(U)


@pytest.mark.parametrize("executor", EXECUTORS)
def test_viscous_stress_ignores_the_order_of_the_patches(executor: str) -> None:
    empty_last, empty_first = (_stress(mesh, executor) for mesh in MESHES)

    np.testing.assert_allclose(
        empty_first,
        empty_last,
        rtol=0,
        atol=ATOL * np.abs(empty_last).max(),
        err_msg="box2D_emptyFirst vs box2D_emptyLast",
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESHES)
def test_viscous_stress_of_a_planar_flow_has_no_z_component(mesh: str, executor: str) -> None:
    stress = _stress(mesh, executor)

    np.testing.assert_allclose(stress[:, 2], 0.0, rtol=0, atol=ATOL, err_msg=mesh)
