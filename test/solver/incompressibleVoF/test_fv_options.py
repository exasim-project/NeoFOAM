# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``fvOptions`` momentum sources in the VoF momentum and pressure equations.

``interFoam``'s ``UEqn.H`` assembles ``… == fvOptions(rho, U)`` — the
*mass-weighted* source overload, which is where the VoF hook differs from the
single-phase one (``test/solver/incompressibleFluid/test_fv_options.py``, which
also pins detection). The weighting is not a magnitude question here but a
dimensional one: the momentum equation carries ``rho*U/dt*V``, so handing it the
single-phase ``fvOptions(U)`` matrix (``U/dt*V``) aborts in ``fvMatrix``'s own
dimension check rather than quietly producing a wrong number — which is what
makes the source magnitude below sufficient evidence that the right overload is
wired. ``pEqn.H`` then calls ``fvOptions.correct(U)`` a second time, after the
velocity reconstruction, and that call is this module's other subject.

**Cases.** Both are ``cases/vofRow4`` geometry re-cut as an open duct — four
cells across a unit cube, split into two blocks so the ``source`` cellZone holds
only the two cells at ``x < 0.5``, with an inlet at ``x = 0`` and a
fixed-pressure outlet at ``x = 1`` so the pressure corrector has an outflow to
balance. ``0/alpha.water`` steps ``0, 0.25, 0.75, 1`` across them.

* ``cases/vofRow4Source`` carries a ``system/fvOptions`` with a
  ``vectorSemiImplicitSource``. With ``volumeMode specific`` the declared value
  is the source per unit volume and ``SemiImplicitSource`` hands it to the
  equation as ``eqn += Su``, i.e. ``source -= V*Su``; the momentum equation then
  *subtracts* that matrix (``==`` is ``-``), so the assembled source must shift
  by exactly ``+V*Su`` on the zone and by nothing outside it.
* ``cases/vofRow4Limited`` carries a ``constant/fvOptions`` (the other of the two
  locations ``fv::options`` searches) with a ``limitVelocity`` correction, which
  contributes nothing to the matrix and acts only through ``correct(U)``,
  scaling ``U`` down to ``max`` in magnitude on the zone. It runs with
  ``momentumPredictor yes``, so both correction points are exercised: the one
  after the momentum solve and the one after the pressure corrector overwrites
  ``U``.

Each case is run twice, in two subprocesses (one ``Foam::Time`` per process), so
both variants start from the pristine ``0/`` state and the difference between the
two velocities is attributable to the fvOptions hooks alone.

Tolerances are ``atol`` only: the quantities are O(1) and ``limitVelocity``
rescales by ``sqrt(max²/|U|²)``, so a couple of ulp is the whole error budget,
while the failure this guards against is the term missing entirely.
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
_CASES = _HERE / "cases"
_WORKER = _HERE / "_fv_options_worker.py"

#: ``system/fvOptions``: ``sources { U ((0 5 0) 0); }``, ``volumeMode specific``.
_SU = np.array([0.0, 5.0, 0.0])

#: ``constant/fvOptions`` of ``vofRow4Limited``: ``max 1``.
_U_MAX = 1.0

#: ``system/blockMeshDict``: the ``source`` block spans ``0 <= x <= 0.5``.
_SOURCE_X_MAX = 0.5

#: A few ulp on an O(1) quantity; see the module docstring.
_ATOL = 1e-14


def _run_case(name: str, tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    """Mesh the case, then run one momentum+continuity pass per variant."""
    case = tmp_path_factory.mktemp(name) / "case"
    shutil.copytree(_CASES / name, case)
    subprocess.run(["blockMesh", "-case", str(case)], check=True, capture_output=True, timeout=300)
    for variant in ("with", "without"):
        subprocess.run(
            [sys.executable, str(_WORKER), str(case), variant],
            check=True,
            capture_output=True,
            timeout=600,
        )
    return {
        variant: json.loads((case / f"fv_options-{variant}.json").read_text())
        for variant in ("with", "without")
    }


@pytest.fixture(scope="module")
def vof_row4_source(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    """Run the semi-implicit-source case both ways."""
    return _run_case("vofRow4Source", tmp_path_factory)


@pytest.fixture(scope="module")
def vof_row4_limited(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    """Run the ``limitVelocity`` case, whose only effect is ``correct(U)``."""
    return _run_case("vofRow4Limited", tmp_path_factory)


def _in_zone(run: dict) -> np.ndarray:
    return np.asarray(run["cell_centres"])[:, 0] < _SOURCE_X_MAX


# --- fvOptions(rho, U) in the momentum equation ----------------------------


def test_the_case_builds_one_option(vof_row4_source: dict[str, dict]) -> None:
    assert vof_row4_source["with"]["options"] == 1


def test_the_momentum_source_gains_the_declared_source_in_the_zone(
    vof_row4_source: dict[str, dict],
) -> None:
    in_zone = _in_zone(vof_row4_source["with"])
    volumes = np.asarray(vof_row4_source["with"]["cell_volumes"])[in_zone, None]

    shift = np.asarray(vof_row4_source["with"]["source"]) - np.asarray(
        vof_row4_source["without"]["source"]
    )
    np.testing.assert_allclose(
        shift[in_zone],
        volumes * _SU,
        rtol=0,
        atol=_ATOL,
        err_msg="vofRow4Source: the momentum source does not carry +V*Su",
    )


def test_the_momentum_source_is_untouched_outside_the_zone(
    vof_row4_source: dict[str, dict],
) -> None:
    outside = ~_in_zone(vof_row4_source["with"])

    # Bit-identical, not merely close: outside its cell zone the source writes
    # nothing, which is the same guarantee that keeps a case with no fvOptions
    # dictionary assembling exactly the matrix it always did.
    np.testing.assert_array_equal(
        np.asarray(vof_row4_source["with"]["source"])[outside],
        np.asarray(vof_row4_source["without"]["source"])[outside],
        err_msg="vofRow4Source: the momentum source changed outside the source zone",
    )


# --- fvOptions.correct(U), after the predictor and after the corrector ------


def test_a_correction_leaves_the_momentum_matrix_alone(
    vof_row4_limited: dict[str, dict],
) -> None:
    # limitVelocity implements neither addSup nor constrain, so the two
    # assemblies must be the same matrix — which is what makes the velocity
    # differences below attributable to correct(U) and to nothing else.
    np.testing.assert_array_equal(
        np.asarray(vof_row4_limited["with"]["source"]),
        np.asarray(vof_row4_limited["without"]["source"]),
        err_msg="vofRow4Limited: a correction changed the momentum matrix",
    )


def test_the_momentum_solve_is_followed_by_the_clamp_inside_the_zone(
    vof_row4_limited: dict[str, dict],
) -> None:
    in_zone = _in_zone(vof_row4_limited["with"])
    unlimited = np.asarray(vof_row4_limited["without"]["U_after_momentum"])[in_zone]
    limited = np.asarray(vof_row4_limited["with"]["U_after_momentum"])[in_zone]

    # Both runs solved the same matrix from the same initial guess, so
    # ``unlimited`` is exactly the velocity the clamp saw.
    assert np.all(np.linalg.norm(unlimited, axis=1) > _U_MAX)
    np.testing.assert_allclose(
        limited,
        unlimited * _U_MAX / np.linalg.norm(unlimited, axis=1)[:, None],
        rtol=0,
        atol=_ATOL,
        err_msg="vofRow4Limited: the solved velocity was not clamped to max",
    )


def test_the_momentum_solve_leaves_the_velocity_outside_the_zone_alone(
    vof_row4_limited: dict[str, dict],
) -> None:
    outside = ~_in_zone(vof_row4_limited["with"])

    np.testing.assert_array_equal(
        np.asarray(vof_row4_limited["with"]["U_after_momentum"])[outside],
        np.asarray(vof_row4_limited["without"]["U_after_momentum"])[outside],
        err_msg="vofRow4Limited: correct(U) reached outside the source zone",
    )


def test_the_pressure_corrector_is_followed_by_the_clamp_inside_the_zone(
    vof_row4_limited: dict[str, dict],
) -> None:
    # pEqn.H's own fvOptions.correct(U): the corrector overwrites U with
    # ``HbyA + rAU*reconstruct(...)``, which knows nothing about the limit, so
    # without the second call the zone would come out of continuity unclamped —
    # as the variant without fvOptions does.
    in_zone = _in_zone(vof_row4_limited["with"])
    uncorrected = np.linalg.norm(
        np.asarray(vof_row4_limited["without"]["U_after_continuity"])[in_zone], axis=1
    )
    corrected = np.linalg.norm(
        np.asarray(vof_row4_limited["with"]["U_after_continuity"])[in_zone], axis=1
    )

    assert np.all(uncorrected > _U_MAX)
    np.testing.assert_allclose(
        corrected,
        _U_MAX,
        rtol=0,
        atol=_ATOL,
        err_msg="vofRow4Limited: the corrected velocity was not clamped to max",
    )
