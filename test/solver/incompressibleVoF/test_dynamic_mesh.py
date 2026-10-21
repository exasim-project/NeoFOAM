# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Moving-mesh support: mesh selection, ``Uf``, the DyM controls, and ``mesh.update()``.

interFoam is a dynamic-mesh solver — it builds its mesh with
``dynamicFvMesh::New`` and calls ``mesh.update()`` at the head of every outer
corrector, rebuilding the buoyancy head ``gh``/``ghf`` whenever the mesh
changed. This module pins the four observable consequences:

* a case *with* ``constant/dynamicMeshDict`` gets that dictionary's motion
  solver, and a case without one keeps the plain static ``fvMesh``;
* the face velocity ``Uf`` (``createUfIfPresent.H``) exists on the moving mesh
  and nowhere else;
* ``createDyMControls.H``'s ``correctPhi`` default is ``mesh.dynamic()``;
* after a run the mesh has actually moved, and ``gh`` matches ``g & C`` on the
  *moved* cell centres rather than the ones it was built on.

**Cases.** ``cases/vofRow4/common`` (static) and the package conftest's
``moving()`` step — the same 4-cell row plus a ``constant/dynamicMeshDict``
oscillating the whole mesh along gravity, a quarter period over the 1 s
``endTime`` so the mesh sits at its full 0.05 m offset when the run ends. The
moving case sets ``correctPhi no``: the
start-up/post-move flux projection (``CorrectPhi``) is a separate, unported part
of interFoam, and 9 of the 17 moving tutorials switch it off too.

The staged-init assertions read the session-scoped pipeline dumps from
``conftest.py``; the run assertion needs a whole time loop and goes through
``_dynamic_mesh_worker.py`` (one ``Foam::Time`` per process).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from .conftest import BuiltCase, build_case, moving

_HERE = Path(__file__).parent
_WORKER = _HERE / "_dynamic_mesh_worker.py"

#: ``cases/vofRow4/common/constant/g``.
_GRAVITY_Y = -9.81

#: ``oscillatingLinearMotionCoeffs``: amplitude 0.05 m along y, omega pi/2, so
#: the offset at the 1 s endTime is a full 0.05 m.
_END_OFFSET_Y = 0.05

#: Undisplaced cell-centre height of the unit-cube row (``system/blockMeshDict``).
_CENTRE_Y0 = 0.5

# --- mesh selection --------------------------------------------------------


@pytest.mark.parametrize(
    "case_fixture, expected_type, expected_dynamic, expected_Uf",
    [
        pytest.param("vof_row4", "fvMesh", False, None, id="no_dynamicMeshDict"),
        pytest.param("vof_row4_moving", "dynamicFvMesh", True, "Uf", id="with_dynamicMeshDict"),
    ],
)
def test_the_dynamicMeshDict_decides_the_mesh_class_and_the_face_velocity(
    request: pytest.FixtureRequest,
    case_fixture: str,
    expected_type: str,
    expected_dynamic: bool,
    expected_Uf: str | None,
) -> None:
    # createUfIfPresent.H builds ``Uf`` on the dynamic mesh and nowhere else.
    built: BuiltCase = request.getfixturevalue(case_fixture)
    assert built.result["mesh_type"] == expected_type
    assert built.result["mesh_dynamic"] is expected_dynamic
    assert built.result["Uf"] == expected_Uf


# --- createDyMControls.H defaults -----------------------------------------


def test_correct_phi_defaults_to_mesh_dynamic_and_the_case_dict_overrides_it(
    vof_row4: BuiltCase, vof_row4_moving: BuiltCase
) -> None:
    # `correctPhi` defaults to mesh.dynamic(), and the other two to False; the
    # moving case then asks for `correctPhi no` even though the mesh is dynamic.
    assert vof_row4.result["dynamic_mesh_controls"] == {
        "correctPhi": False,
        "checkMeshCourantNo": False,
        "moveMeshOuterCorrectors": False,
    }
    assert vof_row4_moving.result["dynamic_mesh_controls"]["correctPhi"] is False


# --- mesh.update() in the time loop ---------------------------------------


@pytest.fixture(scope="module")
def moved_mesh(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the whole time loop on the moving case; return the final mesh state."""
    case = build_case(tmp_path_factory.mktemp("vofRow4Moved") / "case", moving())
    subprocess.run(["blockMesh", "-case", str(case)], check=True, capture_output=True, timeout=300)
    subprocess.run(
        [sys.executable, str(_WORKER), str(case)], check=True, capture_output=True, timeout=600
    )
    return json.loads((case / "dynamic_mesh.json").read_text())


def test_the_time_loop_moves_the_mesh_and_rebuilds_the_buoyancy_head(
    moved_mesh: dict,
) -> None:
    # Without mesh.update() the points never change and polyMesh::moving() stays
    # False; with it the whole row sits one amplitude above where it started.
    assert moved_mesh["moving"] is True
    centres = np.asarray(moved_mesh["cell_centres"])
    np.testing.assert_allclose(
        centres[:, 1],
        _CENTRE_Y0 + _END_OFFSET_Y,
        rtol=0,
        atol=1e-9,
        err_msg="moving: cell centres are not at the end-of-run mesh offset",
    )

    # gh = (g & C) - ghRef, with ghRef = 0 (no constant/hRef). Rebuilt on every
    # mesh change, so at the end of the run it must be the *moved* centres' head
    # — a gh left at its t=0 value would read -9.81*0.5.
    np.testing.assert_allclose(
        np.asarray(moved_mesh["gh"]),
        _GRAVITY_Y * centres[:, 1],
        rtol=1e-12,
        atol=0,
        err_msg="moving: gh was not rebuilt on the moved cell centres",
    )
