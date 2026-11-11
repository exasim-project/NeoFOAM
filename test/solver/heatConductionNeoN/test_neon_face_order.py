# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The order contract: NeoN's faces of a patch are OpenFOAM's, in OpenFOAM's order.

A coupled instance hands its arrays over in mesh order, and the geometry of its
location comes from the backend: on NeoN from the boundary mesh, flat in patch
order. Region ``right`` of ``cases/coupledBlocks``, uncoupled by the overlay
``cases/uncoupledRight`` (blockMesh, 6 x 5 cells,
patches of 5 and 6 faces and an empty one) is read once by NeoN and by OpenFOAM
in one worker process; NeoN copies OpenFOAM's geometry, so centres, areas and
volumes agree to round-off (1e-12 m on a unit block), face by face.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from solver.heatConductionNeoN._run_case import mesh_copy, result

ATOL = 1e-12


@pytest.fixture(scope="module")
def geometry(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    case: Path = mesh_copy(
        "coupledBlocks", tmp_path_factory.mktemp("order") / "case", "uncoupledRight"
    )
    return result(case / "right", "geometry")


@pytest.mark.parametrize("patch", ["outer", "interface", "walls", "frontAndBack"])
@pytest.mark.parametrize("quantity", ["centres", "areas"])
def test_neon_faces_of_a_patch_are_openfoam_s_in_order(
    geometry: dict[str, Any], patch: str, quantity: str
) -> None:
    faces = geometry["patches"][patch]

    np.testing.assert_allclose(
        np.reshape(faces["neon"][quantity], np.shape(faces["foam"][quantity])),
        faces["foam"][quantity],
        rtol=0,
        atol=ATOL,
        err_msg=f"coupledBlocks right patch {patch} face {quantity}",
    )


@pytest.mark.parametrize("quantity", ["centres", "volumes"])
def test_neon_cells_are_openfoam_s_in_order(geometry: dict[str, Any], quantity: str) -> None:
    cells = geometry["cells"]

    np.testing.assert_allclose(
        cells["neon"][quantity],
        cells["foam"][quantity],
        rtol=0,
        atol=ATOL,
        err_msg=f"coupledBlocks right cell {quantity}",
    )
