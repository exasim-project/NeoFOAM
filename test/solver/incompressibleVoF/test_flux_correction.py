# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The flux projection: ``initCorrectPhi.H`` at start-up and ``correctPhi.H`` after a move.

interFoam projects the face flux onto a divergence-free field in two places, and
this module pins both:

* **before the time loop**, unconditionally — ``initCorrectPhi.H`` has no
  enclosing ``if``, and the ``correctPhi`` key only chooses which of two
  equivalent ``rAUf`` forms is passed. A case that never mentions ``correctPhi``
  is projected just the same, which is exactly the group of tutorials that needs
  it most (``waterChannel``, ``iobasin``, ``nozzleFlow2D``, …);
* **inside the outer corrector**, after every ``mesh.update()`` that changed the
  mesh, but only when the case asks for ``correctPhi``.

**Cases.** ``cases/vofRow4Divergent`` is the 4-cell unit-cube row of
``cases/vofRow4`` with an inlet that supplies 1 m/s into an interior carrying
2 m/s, so ``createPhi(U) = fvc::flux(U)`` is *not* solenoidal (see the
derivation on the test). Its PIMPLE dict never mentions ``correctPhi`` — that is
the point. ``cases/vofRow4Moving`` (``correctPhi no``) and
``cases/vofRow4MovingCorrectPhi`` (``correctPhi yes``) are the moving twins;
they differ in that one key alone.

The start-up assertion reads the session-scoped staged-init dump from
``conftest.py``. The two moving assertions need whole time loops, so they go
through ``_dynamic_mesh_worker.py`` (one ``Foam::Time`` per process) and count
``pcorr`` solves in its captured output — every ``fvMatrix::solve`` prints one
``Solving for pcorr`` line, so the count *is* the number of projections. The
worker is reused as-is: it runs the solver, which is all that is needed here.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from .conftest import BuiltCase

_HERE = Path(__file__).parent
_MOVING_CASE = _HERE / "cases" / "vofRow4Moving"
_MOVING_CORRECT_PHI_CASE = _HERE / "cases" / "vofRow4MovingCorrectPhi"
_WORKER = _HERE / "_dynamic_mesh_worker.py"

#: ``system/controlDict``: endTime 1, deltaT 0.25, ``adjustTimeStep no``.
_TIME_STEPS = 4


def _pcorr_solves(case: Path, tmp_path_factory: pytest.TempPathFactory, name: str) -> int:
    """Run *case* through the solver and count its ``pcorr`` solves."""
    run_dir = tmp_path_factory.mktemp(name) / "case"
    shutil.copytree(case, run_dir)
    subprocess.run(
        ["blockMesh", "-case", str(run_dir)], check=True, capture_output=True, timeout=300
    )
    proc = subprocess.run(
        [sys.executable, str(_WORKER), str(run_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc.stdout.count("Solving for pcorr")


# --- initCorrectPhi.H: before the time loop, whatever correctPhi says ------


def test_the_startup_projection_runs_on_a_case_that_never_sets_correct_phi(
    vof_row4_divergent: BuiltCase,
) -> None:
    """A case with ``correctPhi`` unset still gets its start-up flux projected.

    ``0/U`` feeds 1 m/s through the unit-area inlet into an interior at 2 m/s,
    so ``createPhi(U)`` is 2 on all three internal faces and 2 out of the far
    wall — twice what is fed in, and ``div(phi) = 4 1/s`` in the first cell.
    ``CorrectPhi`` first balances the outflow (``adjustPhi``: the zero-gradient
    wall is the only adjustable patch, so its 2 is scaled to 1) and then solves
    ``pcorr``; the only solenoidal flux left in a 1-D duct fed 1 m/s over unit
    area is 1 on every internal face.

    ``rtol`` is the ``pcorr`` solver tolerance from ``system/fvSolution``; the
    unprojected value this has to tell apart is 2.
    """
    assert vof_row4_divergent.result["dynamic_mesh_controls"]["correctPhi"] is False
    np.testing.assert_allclose(
        vof_row4_divergent.internal("phi"),
        [1.0, 1.0, 1.0],
        rtol=1e-5,
        atol=0,
        err_msg="vofRow4Divergent: the start-up projection did not close the flux",
    )


# --- correctPhi.H: after every mesh move, only when the case asks ----------


def test_a_moving_mesh_reprojects_the_flux_after_every_move_when_correct_phi_is_set(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    # One projection before the time loop plus one per time step: the mesh moves
    # on every step (solidBody oscillation) and `moveMeshOuterCorrectors` is off,
    # so `mesh.update()` runs once per step.
    assert (
        _pcorr_solves(_MOVING_CORRECT_PHI_CASE, tmp_path_factory, "vofRow4MovingCorrectPhi")
        == 1 + _TIME_STEPS
    )


def test_a_moving_mesh_projects_only_at_start_up_when_correct_phi_is_off(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    # Same case, `correctPhi no`: the mesh still moves every step, but native
    # skips the re-projection, so only initCorrectPhi.H's single solve is left.
    assert _pcorr_solves(_MOVING_CASE, tmp_path_factory, "vofRow4Moving") == 1
