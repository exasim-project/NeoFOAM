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

**Cases.** ``cases/vofRow4/common`` (static) and its ``moving`` overlay — the
same 4-cell row plus a ``constant/dynamicMeshDict`` oscillating the whole mesh
along gravity, a quarter period over the 1 s ``endTime`` so the mesh sits at its
full 0.05 m offset when the run ends. The moving case sets ``correctPhi no``: the
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

from .conftest import VOF_ROW4, BuiltCase, stage_case

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
    "case_fixture, expected_type, expected_dynamic",
    [
        pytest.param("vof_row4", "fvMesh", False, id="no_dynamicMeshDict"),
        pytest.param("vof_row4_moving", "dynamicFvMesh", True, id="with_dynamicMeshDict"),
    ],
)
def test_the_dynamicMeshDict_decides_which_mesh_class_is_built(
    request: pytest.FixtureRequest,
    case_fixture: str,
    expected_type: str,
    expected_dynamic: bool,
) -> None:
    built: BuiltCase = request.getfixturevalue(case_fixture)
    assert built.result["mesh_type"] == expected_type
    assert built.result["mesh_dynamic"] is expected_dynamic


# --- Uf (createUfIfPresent.H) ---------------------------------------------


@pytest.mark.parametrize(
    "case_fixture, expected_Uf",
    [
        pytest.param("vof_row4", None, id="static_mesh"),
        pytest.param("vof_row4_moving", "Uf", id="dynamic_mesh"),
    ],
)
def test_the_face_velocity_is_built_on_a_dynamic_mesh_and_nowhere_else(
    request: pytest.FixtureRequest, case_fixture: str, expected_Uf: str | None
) -> None:
    built: BuiltCase = request.getfixturevalue(case_fixture)
    assert built.result["Uf"] == expected_Uf


# --- createDyMControls.H defaults -----------------------------------------


def test_correct_phi_defaults_to_off_on_a_static_mesh(vof_row4: BuiltCase) -> None:
    # `correctPhi` defaults to mesh.dynamic(), and the other two to False.
    assert vof_row4.result["dynamic_mesh_controls"] == {
        "correctPhi": False,
        "checkMeshCourantNo": False,
        "moveMeshOuterCorrectors": False,
    }


def test_the_case_dictionary_overrides_the_correct_phi_default(
    vof_row4_moving: BuiltCase,
) -> None:
    # The moving case asks for `correctPhi no` even though the mesh is dynamic.
    assert vof_row4_moving.result["dynamic_mesh_controls"]["correctPhi"] is False


# --- mesh.update() in the time loop ---------------------------------------


@pytest.fixture(scope="module")
def moved_mesh(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the whole time loop on the moving case; return the final mesh state."""
    case = stage_case(
        tmp_path_factory.mktemp("vofRow4Moved") / "case", VOF_ROW4 / "common", VOF_ROW4 / "moving"
    )
    subprocess.run(["blockMesh", "-case", str(case)], check=True, capture_output=True, timeout=300)
    subprocess.run(
        [sys.executable, str(_WORKER), str(case)], check=True, capture_output=True, timeout=600
    )
    return json.loads((case / "dynamic_mesh.json").read_text())


def test_the_time_loop_moves_the_mesh(moved_mesh: dict) -> None:
    # Without mesh.update() the points never change and polyMesh::moving() stays
    # False; with it the whole row sits one amplitude above where it started.
    assert moved_mesh["moving"] is True
    centres_y = np.asarray(moved_mesh["cell_centres"])[:, 1]
    np.testing.assert_allclose(
        centres_y,
        _CENTRE_Y0 + _END_OFFSET_Y,
        rtol=0,
        atol=1e-9,
        err_msg="moving: cell centres are not at the end-of-run mesh offset",
    )


def test_the_buoyancy_head_follows_the_moved_cell_centres(moved_mesh: dict) -> None:
    # gh = (g & C) - ghRef, with ghRef = 0 (no constant/hRef). Rebuilt on every
    # mesh change, so at the end of the run it must be the *moved* centres' head
    # — a gh left at its t=0 value would read -9.81*0.5.
    gh = np.asarray(moved_mesh["gh"])
    centres = np.asarray(moved_mesh["cell_centres"])
    np.testing.assert_allclose(
        gh,
        _GRAVITY_Y * centres[:, 1],
        rtol=1e-12,
        atol=0,
        err_msg="moving: gh was not rebuilt on the moved cell centres",
    )
