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

**Cases.** The ``cases/row4`` base — four unit cells in a row along x, the first
two in a ``modelZone`` cellZone, with ``0/U`` a uniform ``(1 0 0)`` (see
``row4_cases.py``) — under the ``mrf`` variant, whose
``constant/MRFProperties`` spins that zone at ``omega 2`` about ``(0 0 1)``
through the origin; and the same variant on top of ``transient``, i.e. driven by
PIMPLE instead of SIMPLE, because the two algorithms carry their own copy of the
hook. Every expectation below is then hand-derivable from
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
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neofoam.mrf import mrf

from .row4_cases import row4

_HERE = Path(__file__).parent
_WORKER = _HERE / "_mrf_worker.py"

#: ``constant/MRFProperties``: ``axis (0 0 1)``, ``omega 2``.
_OMEGA = np.array([0.0, 0.0, 2.0])

#: ``system/blockMeshDict``: the ``modelZone`` block spans ``0 <= x <= 2``.
_ROTOR_X_MAX = 2.0

#: A few ulp on an O(1) source; see the module docstring.
_ATOL = 1e-14


@pytest.fixture(
    scope="module",
    params=[("mrf",), ("transient", "mrf")],
    ids=["steady_simple", "transient_pimple"],
)
def rotor_row4(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Mesh the case and assemble its momentum equation both ways.

    Parametrized over the two algorithms that own a momentum hook: the ``mrf``
    variant on the steady SIMPLE base, and the same variant on top of
    ``transient`` (``Euler`` ddt, a ``PIMPLE`` dict). Both expectations below are
    the same numbers — the frame terms do not depend on the time derivative — so
    the extra coverage is a parametrize entry, not a test body.
    """
    case = row4(*request.param).build_at(tmp_path_factory.mktemp("mrfRow4") / "case").path
    subprocess.run(["blockMesh", "-case", str(case)], check=True, capture_output=True, timeout=300)
    subprocess.run(
        [sys.executable, str(_WORKER), str(case)], check=True, capture_output=True, timeout=600
    )
    return json.loads((case / "mrf.json").read_text())


# --- detection -------------------------------------------------------------


@pytest.mark.parametrize(
    ("variants", "active"),
    # The bare base is the same case with only the dictionary missing: nothing
    # else about a case may switch MRF on, because an active model is what
    # changes the assembly.
    [(("mrf",), True), ((), False)],
    ids=["with_MRFProperties", "without_MRFProperties"],
)
def test_only_constant_mrf_properties_activates_the_model(
    variants: tuple[str, ...], active: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(row4(*variants).build_at(tmp_path / "case").path)

    assert mrf.run_detect() is active


# --- MRF.correctBoundaryVelocity(U) ---------------------------------------


def test_only_the_rotating_wall_faces_get_the_frame_velocity(rotor_row4: dict) -> None:
    centres = np.asarray(rotor_row4["walls_face_centres"])
    in_zone = centres[:, 0] < _ROTOR_X_MAX
    outside = centres[:, 0] > _ROTOR_X_MAX
    after = np.asarray(rotor_row4["walls_U_after"])

    np.testing.assert_allclose(
        after[in_zone],
        np.cross(_OMEGA, centres[in_zone]),
        rtol=0,
        atol=_ATOL,
        err_msg="row4+mrf: rotating wall faces do not carry Omega x Cf",
    )
    # Outside the zone the faces keep the no-slip value they came in with.
    np.testing.assert_array_equal(
        after[outside],
        np.asarray(rotor_row4["walls_U_before"])[outside],
        err_msg="row4+mrf: correctBoundaryVelocity reached outside the modelZone",
    )


# --- MRF.DDt(U) in the momentum equation ----------------------------------


def test_the_momentum_source_gains_the_coriolis_term_in_the_zone_only(rotor_row4: dict) -> None:
    centres = np.asarray(rotor_row4["cell_centres"])
    in_zone = centres[:, 0] < _ROTOR_X_MAX
    outside = centres[:, 0] > _ROTOR_X_MAX
    volumes = np.asarray(rotor_row4["cell_volumes"])[in_zone, None]

    # Against the assembly on the *same* corrected boundary state, so the shift
    # is the frame acceleration and nothing else.
    with_mrf = np.asarray(rotor_row4["source_with_mrf"])
    without_mrf = np.asarray(rotor_row4["source_without_mrf_corrected_walls"])
    # U is the uniform (1 0 0) of 0/U in every cell, so Omega x U is one vector.
    np.testing.assert_allclose(
        (with_mrf - without_mrf)[in_zone],
        -volumes * np.cross(_OMEGA, [1.0, 0.0, 0.0]),
        rtol=0,
        atol=_ATOL,
        err_msg="row4+mrf: the momentum source does not carry -V*(Omega x U)",
    )
    # Outside the zone bit-identical, not merely close: there ``addCoriolis``
    # writes nothing, which is the same guarantee that keeps a case with no
    # ``MRFProperties`` assembling exactly the matrix it always did.
    np.testing.assert_array_equal(
        with_mrf[outside],
        without_mrf[outside],
        err_msg="row4+mrf: the momentum source changed outside the modelZone",
    )
