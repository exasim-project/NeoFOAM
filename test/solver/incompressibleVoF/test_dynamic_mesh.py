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

**Cases.** ``cases/vofRow4`` (static) and ``cases/vofRow4Moving`` — the same
4-cell row plus a ``constant/dynamicMeshDict`` oscillating the whole mesh along
gravity, a quarter period over the 1 s ``endTime`` so the mesh sits at its full
0.05 m offset when the run ends. The moving case sets ``correctPhi no``: the
start-up/post-move flux projection (``CorrectPhi``) is a separate, unported part
of interFoam, and 9 of the 17 moving tutorials switch it off too.

The staged-init assertions read the session-scoped pipeline dumps from
``conftest.py``; the run assertion needs a whole time loop and goes through
``_dynamic_mesh_worker.py`` (one ``Foam::Time`` per process).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from .conftest import BuiltCase

_HERE = Path(__file__).parent
_MOVING_CASE = _HERE / "cases" / "vofRow4Moving"
_WORKER = _HERE / "_dynamic_mesh_worker.py"

#: ``cases/vofRow4Moving/constant/g``.
_GRAVITY_Y = -9.81

#: ``oscillatingLinearMotionCoeffs``: amplitude 0.05 m along y, omega pi/2, so
#: the offset at the 1 s endTime is a full 0.05 m.
_END_OFFSET_Y = 0.05

#: Undisplaced cell-centre height of the unit-cube row (``system/blockMeshDict``).
_CENTRE_Y0 = 0.5


# --- mesh selection --------------------------------------------------------


def test_a_case_without_a_dynamicMeshDict_keeps_the_static_mesh(
    vof_row4: BuiltCase,
) -> None:
    assert vof_row4.result["mesh_type"] == "fvMesh"
    assert vof_row4.result["mesh_dynamic"] is False


def test_a_case_with_a_dynamicMeshDict_gets_its_motion_solver(
    vof_row4_moving: BuiltCase,
) -> None:
    assert vof_row4_moving.result["mesh_type"] == "dynamicFvMesh"
    assert vof_row4_moving.result["mesh_dynamic"] is True


# --- Uf (createUfIfPresent.H) ---------------------------------------------


def test_no_face_velocity_is_built_on_a_static_mesh(vof_row4: BuiltCase) -> None:
    assert vof_row4.result["Uf"] is None


def test_the_face_velocity_is_built_on_a_dynamic_mesh(
    vof_row4_moving: BuiltCase,
) -> None:
    assert vof_row4_moving.result["Uf"] == "Uf"


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
    case = tmp_path_factory.mktemp("vofRow4Moved") / "case"
    shutil.copytree(_MOVING_CASE, case)
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
        err_msg="vofRow4Moving: cell centres are not at the end-of-run mesh offset",
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
        err_msg="vofRow4Moving: gh was not rebuilt on the moved cell centres",
    )
