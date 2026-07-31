# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MRF (rotating reference frame) zones: detection, wall velocity, Coriolis term.

``simpleFoam``/``pimpleFoam``'s ``UEqn.H`` opens with
``MRF.correctBoundaryVelocity(U)`` and carries ``+ MRF.DDt(U)`` in the momentum
sum. Both are hooked into each algorithm's ``momentum`` operation behind an
optional injected ``mrf_zones``, which the solver only puts on the Context when
the case has ``constant/MRFProperties`` —
so this module pins three things: the model activates on that file and nothing
else, the rotating patch faces pick up the frame wall velocity, and the momentum
source gains the frame acceleration *inside the rotating cell zone only*.

**Cases.** ``cases/rotorRow4`` — four unit cells in a row along x, the first two
in a ``rotor`` cellZone that ``constant/MRFProperties`` spins at
``omega 2`` about ``(0 0 1)`` through the origin, with ``0/U`` a uniform
``(1 0 0)`` — and ``cases/rotorRow4Transient``, the same case driven by PIMPLE
instead of SIMPLE, because the two algorithms carry their own copy of the hook.
Every expectation below is then hand-derivable from
``MRFZone::addCoriolis`` / ``MRFZone::correctBoundaryVelocity``:

* wall velocity on a zone face:  ``Omega x Cf`` = ``(-2*Cf_y, 2*Cf_x, 0)``
* frame acceleration in a zone cell: ``Omega x U`` = ``(0, 2, 0)``
* an ``fvMatrix`` takes an explicit vector field into its source as
  ``source -= V*field``, and the cells are unit volume, so the momentum source
  must shift by exactly ``(0, -2, 0)`` — and by nothing at all outside the zone.

``momentumPredictor no`` keeps the operation to assembly, so every matrix in
``_mrf_worker.py`` is built on the same ``U``; the wall velocities below are the
ones the ``momentum`` call itself left behind, and the Coriolis shift is taken
between two assemblies that share that corrected boundary state. Tolerances are
``atol`` only: the quantities are exact in binary, and the comparison is a
difference of two O(1)-magnitude sources, so a few ulp is the whole error budget.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neofoam.mrf import mrf

_HERE = Path(__file__).parent
_CASES = _HERE / "cases"
_CASE = _CASES / "rotorRow4"
_WORKER = _HERE / "_mrf_worker.py"

#: ``constant/MRFProperties``: ``axis (0 0 1)``, ``omega 2``.
_OMEGA = np.array([0.0, 0.0, 2.0])

#: ``system/blockMeshDict``: the ``rotor`` block spans ``0 <= x <= 2``.
_ROTOR_X_MAX = 2.0

#: A few ulp on an O(1) source; see the module docstring.
_ATOL = 1e-14


@pytest.fixture(scope="module", params=["rotorRow4", "rotorRow4Transient"])
def rotor_row4(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Mesh the case and assemble its momentum equation both ways.

    Parametrized over the two algorithms that own a momentum hook: ``rotorRow4``
    is the steady SIMPLE case, ``rotorRow4Transient`` its PIMPLE twin (``Euler``
    ddt, a ``PIMPLE`` dict). Both expectations below are the same numbers — the
    frame terms do not depend on the time derivative — so the extra coverage is
    a case directory, not a test body.
    """
    case = tmp_path_factory.mktemp(request.param) / "case"
    shutil.copytree(_CASES / request.param, case)
    subprocess.run(["blockMesh", "-case", str(case)], check=True, capture_output=True, timeout=300)
    subprocess.run(
        [sys.executable, str(_WORKER), str(case)], check=True, capture_output=True, timeout=600
    )
    return json.loads((case / "mrf.json").read_text())


# --- detection -------------------------------------------------------------


def test_the_model_activates_on_a_case_with_mrf_properties(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)
    monkeypatch.chdir(case)

    assert mrf.run_detect() is True


def test_the_model_stays_inactive_without_mrf_properties(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same case with only the dictionary removed: nothing else about a case
    # may switch MRF on, because an active model is what changes the assembly.
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)
    (case / "constant" / "MRFProperties").unlink()
    monkeypatch.chdir(case)

    assert mrf.run_detect() is False


# --- MRF.correctBoundaryVelocity(U) ---------------------------------------


def test_the_rotating_wall_faces_get_the_frame_velocity(rotor_row4: dict) -> None:
    centres = np.asarray(rotor_row4["walls_face_centres"])
    in_zone = centres[:, 0] < _ROTOR_X_MAX

    np.testing.assert_allclose(
        np.asarray(rotor_row4["walls_U_after"])[in_zone],
        np.cross(_OMEGA, centres[in_zone]),
        rtol=0,
        atol=_ATOL,
        err_msg="rotorRow4: rotating wall faces do not carry Omega x Cf",
    )


def test_the_wall_faces_outside_the_zone_keep_their_no_slip_value(rotor_row4: dict) -> None:
    centres = np.asarray(rotor_row4["walls_face_centres"])
    outside = centres[:, 0] > _ROTOR_X_MAX

    before = np.asarray(rotor_row4["walls_U_before"])[outside]
    after = np.asarray(rotor_row4["walls_U_after"])[outside]
    np.testing.assert_array_equal(
        after,
        before,
        err_msg="rotorRow4: correctBoundaryVelocity reached outside the rotor zone",
    )


# --- MRF.DDt(U) in the momentum equation ----------------------------------


def test_the_momentum_source_gains_the_coriolis_term_in_the_zone(rotor_row4: dict) -> None:
    centres = np.asarray(rotor_row4["cell_centres"])
    in_zone = centres[:, 0] < _ROTOR_X_MAX
    volumes = np.asarray(rotor_row4["cell_volumes"])[in_zone, None]

    # Against the assembly on the *same* corrected boundary state, so the shift
    # is the frame acceleration and nothing else.
    shift = np.asarray(rotor_row4["source_with_mrf"]) - np.asarray(
        rotor_row4["source_without_mrf_corrected_walls"]
    )
    # U is the uniform (1 0 0) of 0/U in every cell, so Omega x U is one vector.
    np.testing.assert_allclose(
        shift[in_zone],
        -volumes * np.cross(_OMEGA, [1.0, 0.0, 0.0]),
        rtol=0,
        atol=_ATOL,
        err_msg="rotorRow4: the momentum source does not carry -V*(Omega x U)",
    )


def test_the_momentum_source_is_untouched_outside_the_zone(rotor_row4: dict) -> None:
    centres = np.asarray(rotor_row4["cell_centres"])
    outside = centres[:, 0] > _ROTOR_X_MAX

    with_mrf = np.asarray(rotor_row4["source_with_mrf"])[outside]
    without_mrf = np.asarray(rotor_row4["source_without_mrf_corrected_walls"])[outside]
    # Bit-identical, not merely close: outside the zone ``addCoriolis`` writes
    # nothing, which is the same guarantee that keeps a case with no
    # ``MRFProperties`` assembling exactly the matrix it always did.
    np.testing.assert_array_equal(
        with_mrf,
        without_mrf,
        err_msg="rotorRow4: the momentum source changed outside the rotor zone",
    )
