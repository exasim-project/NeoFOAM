# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``fvOptions`` momentum sources: detection, the source term, and the corrections.

``simpleFoam``/``pimpleFoam``'s ``UEqn.H`` assembles ``… == fvOptions(U)``,
relaxes, then applies ``fvOptions.constrain(UEqn)``, and calls
``fvOptions.correct(U)`` after the momentum solve; ``pEqn.H`` calls it once more
after the velocity corrector. All of that is hooked into each algorithm's
``momentum``/``continuity`` operations behind an optional injected
``fv_options``, which the solver only puts on the Context when the case has
``constant/fvOptions`` or ``system/fvOptions`` — so this module pins four
things: the model activates on either of those files and nothing else, the
declared source lands in the momentum matrix *and nowhere outside its cell zone*,
the correction is applied to ``U`` *after* the momentum solve rather than before
it, and it is applied *again* after the pressure corrector overwrites ``U``.

**Cases.** All three are the ``cases/row4`` base — four unit cells in a row along
x, the first two in a ``modelZone`` cellZone (see ``row4_cases.py``, and
``test_mrf.py`` for the other model that uses it) — plus a variant:

* the ``fvOptions`` variant, alone and on top of the ``transient`` one, carries a
  ``system/fvOptions`` with a ``vectorSemiImplicitSource``. Both algorithms own
  their own copy of the hook, hence the second composition. With
  ``volumeMode specific`` the declared value is the source per unit volume, and
  ``SemiImplicitSource`` hands it to the equation as ``eqn += Su``, i.e.
  ``source -= V*Su``; the momentum equation then *subtracts* that matrix
  (``==`` is ``-``), so the assembled source must shift by exactly ``+V*Su``.
* the ``fvOptionsLimit`` variant carries a ``constant/fvOptions`` (the other of
  the two locations ``fv::options`` searches) with a ``limitVelocity``
  correction, which contributes nothing to the matrix and acts only through
  ``correct(U)``, scaling ``U`` down to ``max`` in magnitude on the zone. Its
  ``0/U`` is ``(3 0 0)`` and ``momentumPredictor`` is on, so the clamp bites on a
  velocity the solve produced — and were the correction called before the solve
  instead of after it, the solve would simply overwrite the clamp.

Each case is run twice, in two subprocesses (one ``Foam::Time`` per process), so
both variants start from the pristine ``0/`` state: same matrix, same
linear-solver initial guess, and the difference between the two velocities is
attributable to the fvOptions hooks alone.

Tolerances are ``atol`` only where a tolerance is needed at all: the source shift
is a difference of two O(1) matrix sources, so a few ulp is the whole error
budget, and the untouched-cell expectations are exact equality because "nothing
was written there" is the guarantee that also keeps a case *without* an
``fvOptions`` dictionary assembling exactly the matrix it always did.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neofoam.fv_options import fvOptions

from .row4_cases import row4

_HERE = Path(__file__).parent
_WORKER = _HERE / "_fv_options_worker.py"

#: ``system/fvOptions``: ``sources { U ((0 5 0) 0); }``, ``volumeMode specific``.
_SU = np.array([0.0, 5.0, 0.0])

#: ``constant/fvOptions`` of the ``fvOptionsLimit`` variant: ``max 1``.
_U_MAX = 1.0

#: ``system/blockMeshDict``: the ``modelZone`` block spans ``0 <= x <= 2``.
_SOURCE_X_MAX = 2.0

#: A few ulp on an O(1) source; see the module docstring.
_ATOL = 1e-14


def _run_case(
    name: str, variants: tuple[str, ...], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, dict]:
    """Compose and mesh the case, then run one momentum+continuity pass per variant."""
    case = row4(*variants).build_at(tmp_path_factory.mktemp(name) / "case").path
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


@pytest.fixture(
    scope="module",
    params=[("fvOptions",), ("transient", "fvOptions")],
    ids=["steady_simple", "transient_pimple"],
)
def sourced_row4(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, dict]:
    """Run the semi-implicit-source case on both algorithms.

    Parametrized over the two that own a momentum hook: the ``fvOptions`` variant
    on the steady SIMPLE base, and the same variant on top of ``transient``
    (``Euler`` ddt, a ``PIMPLE`` dict). Both expectations below are the same
    numbers — an explicit source does not depend on the time derivative — so the
    extra coverage is a parametrize entry, not a test body.
    """
    return _run_case("sourcedRow4", request.param, tmp_path_factory)


@pytest.fixture(scope="module")
def limited_row4(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    """Run the ``limitVelocity`` case, whose only effect is ``correct(U)``."""
    return _run_case("limitedRow4", ("fvOptionsLimit",), tmp_path_factory)


def _in_zone(run: dict) -> np.ndarray:
    return np.asarray(run["cell_centres"])[:, 0] < _SOURCE_X_MAX


# --- detection -------------------------------------------------------------


@pytest.mark.parametrize(
    ("variants", "active"),
    [
        # One variant keeps the dictionary in system/, the other in constant/ — the
        # two locations fv::options searches, and both must switch the model on.
        (("fvOptions",), True),
        (("fvOptionsLimit",), True),
        # The bare base is the same case with only the dictionary missing (from
        # either location): nothing else about a case may switch fvOptions on,
        # because an active model is what changes the assembly.
        ((), False),
    ],
    ids=["system_fvOptions", "constant_fvOptions", "no_fvOptions"],
)
def test_only_an_fvoptions_dictionary_activates_the_model(
    variants: tuple[str, ...], active: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(row4(*variants).build_at(tmp_path / "case").path)

    assert fvOptions.run_detect() is active


# --- fvOptions(U) in the momentum equation ---------------------------------


def test_the_case_builds_one_option(sourced_row4: dict[str, dict]) -> None:
    assert sourced_row4["with"]["options"] == 1


def test_the_momentum_source_gains_the_declared_source_in_the_zone_only(
    sourced_row4: dict[str, dict],
) -> None:
    in_zone = _in_zone(sourced_row4["with"])
    volumes = np.asarray(sourced_row4["with"]["cell_volumes"])[in_zone, None]
    with_source = np.asarray(sourced_row4["with"]["source"])
    without_source = np.asarray(sourced_row4["without"]["source"])

    np.testing.assert_allclose(
        (with_source - without_source)[in_zone],
        volumes * _SU,
        rtol=0,
        atol=_ATOL,
        err_msg="row4+fvOptions: the momentum source does not carry +V*Su",
    )
    # Outside the zone bit-identical, not merely close: there the source writes
    # nothing, which is the same guarantee that keeps a case with no fvOptions
    # dictionary assembling exactly the matrix it always did.
    np.testing.assert_array_equal(
        with_source[~in_zone],
        without_source[~in_zone],
        err_msg="row4+fvOptions: the momentum source changed outside the source zone",
    )


# --- fvOptions.correct(U), after the momentum solve and after the corrector --


def test_a_correction_leaves_the_momentum_matrix_alone(limited_row4: dict[str, dict]) -> None:
    # limitVelocity implements neither addSup nor constrain, so the two
    # assemblies must be the same matrix — which is what makes the velocity
    # differences below attributable to correct(U) and to nothing else.
    np.testing.assert_array_equal(
        np.asarray(limited_row4["with"]["source"]),
        np.asarray(limited_row4["without"]["source"]),
        err_msg="row4+fvOptionsLimit: a correction changed the momentum matrix",
    )


def test_the_momentum_solve_is_followed_by_the_clamp_inside_the_zone_only(
    limited_row4: dict[str, dict],
) -> None:
    in_zone = _in_zone(limited_row4["with"])
    unlimited = np.asarray(limited_row4["without"]["U_after_momentum"])
    limited = np.asarray(limited_row4["with"]["U_after_momentum"])

    # Every zone cell leaves the solve above the limit, so limitVelocity scales
    # each one onto the limiting sphere: U*max/|U|. Both runs solved the same
    # matrix from the same initial guess, so ``unlimited`` is exactly what the
    # clamp saw.
    assert np.all(np.linalg.norm(unlimited[in_zone], axis=1) > _U_MAX)
    np.testing.assert_allclose(
        limited[in_zone],
        unlimited[in_zone] * _U_MAX / np.linalg.norm(unlimited[in_zone], axis=1)[:, None],
        rtol=0,
        atol=_ATOL,
        err_msg="row4+fvOptionsLimit: the solved velocity was not clamped to max",
    )
    np.testing.assert_array_equal(
        limited[~in_zone],
        unlimited[~in_zone],
        err_msg="row4+fvOptionsLimit: correct(U) reached outside the source zone",
    )


def test_the_pressure_corrector_is_followed_by_the_clamp_inside_the_zone(
    limited_row4: dict[str, dict],
) -> None:
    # pEqn.H's own fvOptions.correct(U): the corrector overwrites U with
    # ``HbyA - rAtU*grad(p)``, which knows nothing about the limit, so without
    # the second call the zone would come out of continuity unclamped — as the
    # variant without fvOptions does.
    in_zone = _in_zone(limited_row4["with"])
    uncorrected = np.linalg.norm(
        np.asarray(limited_row4["without"]["U_after_continuity"])[in_zone], axis=1
    )
    corrected = np.linalg.norm(
        np.asarray(limited_row4["with"]["U_after_continuity"])[in_zone], axis=1
    )

    assert np.all(uncorrected > _U_MAX)
    np.testing.assert_allclose(
        corrected,
        _U_MAX,
        rtol=0,
        atol=_ATOL,
        err_msg="row4+fvOptionsLimit: the corrected velocity was not clamped to max",
    )
