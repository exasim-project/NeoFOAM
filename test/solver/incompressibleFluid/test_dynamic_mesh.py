# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Moving-mesh support: mesh selection, ``Uf``, the DyM controls, and the flux chain.

``pimpleFoam`` is a moving-mesh solver — it builds its mesh with
``dynamicFvMesh::New``, calls ``mesh.controlledUpdate()`` at the head of every
outer corrector, and closes ``pEqn.H`` on ``fvc::correctUf`` +
``fvc::makeRelative``, so the flux it carries into the next step is relative to
the mesh motion. This module pins those consequences on a case where every
number is derivable by hand.

**Case.** ``cases/movingRow4`` — a duct of four unit cells along x with a uniform
``(1 0 0)`` inflow and ``slip`` side walls, whose ``constant/dynamicMeshDict``
slides the whole mesh along x at a constant ``0.25 m/s`` (``linearMotion``, i.e.
displacement ``velocity*t`` exactly). The static twin is the same case with only
that dictionary deleted, which is the sole thing that may switch mesh motion on.

The solution stays the uniform ``(1 0 0)`` it starts from — a uniform field with
consistent boundary conditions leaves every term of the momentum equation zero —
so on unit faces:

* the *absolute* face flux is ``1``;
* the mesh flux is the swept volume per step, ``area*velocity`` = ``0.25``;
* hence the flux the solver carries, ``phi``, is ``1 - 0.25 = 0.75`` on a moving
  mesh and stays ``1`` on the static twin;
* ``Uf`` is the face velocity ``(1 0 0)`` — ``fvc::correctUf`` reproduces it from
  the corrected ``U``/``phi`` only if the relative/absolute bookkeeping is right.

Tolerances are ``atol`` only: the quantities are exact in binary arithmetic and
the pressure solves converge to a zero solution, so a few ulp is the whole error
budget.

The run needs a whole time loop and goes through ``_dynamic_mesh_worker.py``
(one ``Foam::Time`` per process).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
_CASE = _HERE / "cases" / "movingRow4"
_WORKER = _HERE / "_dynamic_mesh_worker.py"

#: ``constant/dynamicMeshDict``: ``linearMotion`` at 0.25 m/s along x.
_MESH_VELOCITY_X = 0.25

#: ``system/controlDict``: the run ends at t = 1 s.
_END_TIME = 1.0

#: ``0/U``: the uniform inflow the whole solution stays at.
_FLOW_VELOCITY_X = 1.0

#: Undisplaced cell-centre x of the four unit cells (``system/blockMeshDict``).
_CENTRES_X0 = np.array([0.5, 1.5, 2.5, 3.5])

#: A few ulp on O(1) fluxes; see the module docstring.
_ATOL = 1e-12


def _run(case: Path) -> dict:
    subprocess.run(["blockMesh", "-case", str(case)], check=True, capture_output=True, timeout=300)
    subprocess.run(
        [sys.executable, str(_WORKER), str(case)], check=True, capture_output=True, timeout=600
    )
    return json.loads((case / "dynamic_mesh.json").read_text())


@pytest.fixture(scope="module")
def moving_row4(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The moving duct, run to its end time."""
    case = tmp_path_factory.mktemp("movingRow4") / "case"
    shutil.copytree(_CASE, case)
    return _run(case)


@pytest.fixture(scope="module")
def static_row4(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The same duct with only ``constant/dynamicMeshDict`` removed."""
    case = tmp_path_factory.mktemp("staticRow4") / "case"
    shutil.copytree(_CASE, case)
    (case / "constant" / "dynamicMeshDict").unlink()
    return _run(case)


# --- mesh selection --------------------------------------------------------


def test_a_case_with_a_dynamicMeshDict_gets_its_motion_solver(moving_row4: dict) -> None:
    assert moving_row4["mesh_type"] == "dynamicFvMesh"
    assert moving_row4["dynamic"] is True


def test_a_case_without_a_dynamicMeshDict_keeps_the_static_mesh(static_row4: dict) -> None:
    assert static_row4["mesh_type"] == "fvMesh"
    assert static_row4["dynamic"] is False


# --- Uf (createUfIfPresent.H) ---------------------------------------------


def test_the_face_velocity_is_built_on_a_dynamic_mesh(moving_row4: dict) -> None:
    faces = np.asarray(moving_row4["Uf"])
    np.testing.assert_allclose(
        faces,
        np.tile([_FLOW_VELOCITY_X, 0.0, 0.0], (len(faces), 1)),
        rtol=0,
        atol=_ATOL,
        err_msg="movingRow4: Uf is not the face velocity fvc::correctUf should leave",
    )


def test_no_face_velocity_is_built_on_a_static_mesh(static_row4: dict) -> None:
    assert static_row4["Uf"] is None


# --- createDyMControls.H defaults -----------------------------------------


def test_correct_phi_defaults_to_on_for_a_moving_mesh(moving_row4: dict) -> None:
    # `correctPhi` defaults to mesh.dynamic(), the other two to False.
    assert moving_row4["dynamic_mesh_controls"] == {
        "correctPhi": True,
        "checkMeshCourantNo": False,
        "moveMeshOuterCorrectors": False,
    }


def test_correct_phi_defaults_to_off_for_a_static_mesh(static_row4: dict) -> None:
    assert static_row4["dynamic_mesh_controls"] == {
        "correctPhi": False,
        "checkMeshCourantNo": False,
        "moveMeshOuterCorrectors": False,
    }


# --- mesh.update() in the time loop ---------------------------------------


def test_the_time_loop_moves_the_mesh(moving_row4: dict) -> None:
    # Without the mesh-motion step the points never change and polyMesh::moving()
    # stays False; with it the duct has slid velocity*endTime downstream.
    assert moving_row4["moving"] is True
    centres_x = np.asarray(moving_row4["cell_centres"])[:, 0]
    np.testing.assert_allclose(
        centres_x,
        _CENTRES_X0 + _MESH_VELOCITY_X * _END_TIME,
        rtol=0,
        atol=_ATOL,
        err_msg="movingRow4: cell centres are not at the end-of-run mesh offset",
    )


def test_the_static_mesh_never_moves(static_row4: dict) -> None:
    assert static_row4["moving"] is False
    np.testing.assert_array_equal(
        np.asarray(static_row4["cell_centres"])[:, 0],
        _CENTRES_X0,
        err_msg="staticRow4: the cell centres moved on a case with no dynamicMeshDict",
    )


# --- the flux the solver carries (fvc::makeRelative) -----------------------


def test_the_flux_is_relative_to_the_mesh_motion(moving_row4: dict) -> None:
    # Unit faces: the absolute flux is the flow velocity and the mesh flux is the
    # mesh velocity, so what pEqn.H hands on is their difference. A run that moved
    # the mesh but skipped makeRelative would read 1.0 here.
    np.testing.assert_allclose(
        np.asarray(moving_row4["phi"]),
        _FLOW_VELOCITY_X - _MESH_VELOCITY_X,
        rtol=0,
        atol=_ATOL,
        err_msg="movingRow4: phi is not the flux relative to the mesh motion",
    )


def test_the_static_flux_is_the_absolute_one(static_row4: dict) -> None:
    np.testing.assert_allclose(
        np.asarray(static_row4["phi"]),
        _FLOW_VELOCITY_X,
        rtol=0,
        atol=_ATOL,
        err_msg="staticRow4: the mesh-motion terms perturbed a static case",
    )
