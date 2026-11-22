# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The closed-domain pressure reference of the VoF PIMPLE algorithm.

incompressibleVoF always *solves* for ``p_rgh`` (interFoam's dynamic pressure);
the absolute pressure ``p`` is only ever reconstructed as ``p_rgh + rho*gh``.
On a **closed** domain that reconstruction leaves the pressure level
undetermined, so interFoam's ``pEqn.H`` pins it::

    p == p_rgh + rho*gh;
    if (p_rgh.needReference())
    {
        p += dimensionedScalar("p", p.dimensions(),
                               pRefValue - getRefCellValue(p, pRefCell));
        p_rgh = p - rho*gh;
    }

``pressure_reference.py`` is the Python composition of that block plus the two
OpenFOAM primitives pybFoam does not bind: ``getRefCellValue``, and the
``needReference()`` answer, which ``setRefCell`` computes but whose ``bool``
return the binding drops — it survives as the sign of the cell index. This
module pins all three.

``createFields.H`` runs the *same* block once at start-up, immediately after
``setRefCell``, so a case whose pressure corrector never runs (``frozenFlow
yes``) still writes a correctly levelled ``p_rgh``. The ``pressure_reference``
init step is where NeoFOAM does it, so the levelling is pinned here twice: on
the fields as init leaves them (``*_before``) and after a further corrector-tail
call (``*_after``).

Two real 4-cell cases, identical apart from their ``0/p_rgh`` boundary
conditions, make the open/closed distinction observable:

* ``cases/closed`` — one ``wall`` patch, ``fixedFluxPressure`` (fixes a
  *gradient*): closed, and built with :data:`CLOSED_REFERENCE`
  (``pRefCell 2; pRefValue 50``). Neither is the default 0, so honouring the
  cell index and the value are separate claims.
* ``cases/open`` — the interFoam damBreak ``atmosphere`` arrangement, a
  ``totalPressure`` patch (a ``fixedValue`` descendant): open, and built with
  :data:`NO_REFERENCE`, which strips ``pRefCell``/``pRefValue`` from the PIMPLE
  dict entirely.

Both are 0/-field overlays of the shared ``cases/vofRow4/common`` base (see the
package conftest) — the unit box cut into 4 cells along x (cell centres at
y = 0.5) with
the interFoam damBreak properties, so every expected value is hand-derivable:

    alpha.water = (0, 0.25, 0.75, 1),  rho = alpha*1000 + (1 - alpha)*1
                                           = (1, 250.75, 750.25, 1000)
    g = (0, -9.81, 0), hRef = 0  ->  gh = g & C = -4.905 in every cell
    p_rgh (from 0/p_rgh)                   = (100, 200, 300, 400)
    p = p_rgh + rho*gh = (95.095, -1029.92875, -3379.97625, -4505)

and for the closed case, with ``pRefValue - p[2] = 50 + 3379.97625``:

    p     = (3525.07125, 2400.0475, 50, -1075.02375)
    p_rgh = p - rho*gh = (3529.97625, 3629.97625, 3729.97625, 3829.97625)

A process owns exactly one ``Foam::Time``, so each case is meshed and run
through ``_pressure_reference_worker.py`` in its own subprocess, once per
session, and the tests read the worker's JSON dump.

Tolerance: the expected values above are exact decimals, but the runtime path
computes them through products of order 5e3 (``rho*gh``), so a handful of ulp
of round-off is expected — ``rtol=1e-13`` is ~50 ulp at that magnitude and
still four orders tighter than any physically meaningful difference. The two
claims that must hold *bit-exactly* (the reference cell lands on ``pRefValue``;
the open-domain block touches nothing) are asserted with ``==``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from numpy.testing import assert_allclose

from neofoam.solver.incompressibleVoF.models.pressure_velocity.pimpleAlgorithm import (
    PimpleFvSchemes,
    PimpleFvSolution,
)
from neofoam.tooling.casebuild import patch

from ...conftest import build_case, overlay

_HERE = Path(__file__).parent
_CASES = _HERE / "cases"
_WORKER = _HERE / "_pressure_reference_worker.py"

# Hand-derived from the case files (see module docstring).
_RHO = [1.0, 250.75, 750.25, 1000.0]
_GH = [-4.905, -4.905, -4.905, -4.905]
_P_RGH_FILE = [100.0, 200.0, 300.0, 400.0]
_P_RECONSTRUCTED = [95.095, -1029.92875, -3379.97625, -4505.0]
_P_LEVELLED = [3525.07125, 2400.0475, 50.0, -1075.02375]
_P_RGH_RELEVELLED = [3529.97625, 3629.97625, 3729.97625, 3829.97625]

_RTOL = 1e-13

#: The closed case's reference, deliberately neither the default 0 cell nor the
#: default 0 value, so honouring the cell index and the value are separate
#: claims. The open case drops both keys: its ``totalPressure`` inlet already
#: pins the level, exactly as the interFoam damBreak tutorial does.
CLOSED_REFERENCE = patch("system/fvSolution", **{"PIMPLE.pRefCell": 2, "PIMPLE.pRefValue": 50})
NO_REFERENCE = patch("system/fvSolution", remove=["PIMPLE.pRefCell", "PIMPLE.pRefValue"])


def _run_worker(case_name: str, tmp_path_factory: pytest.TempPathFactory) -> Any:
    case = build_case(
        tmp_path_factory.mktemp(case_name) / "case",
        overlay(_CASES / case_name),
        CLOSED_REFERENCE if case_name == "closed" else NO_REFERENCE,
    )
    subprocess.run(
        ["blockMesh", "-case", str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    subprocess.run(
        [sys.executable, str(_WORKER), str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return json.loads((case / "pressure_reference.json").read_text())


@pytest.fixture(scope="session")
def closed(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return _run_worker("closed", tmp_path_factory)


@pytest.fixture(scope="session")
def open_domain(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return _run_worker("open", tmp_path_factory)


# --------------------------------------------------------------------------- #
# need_reference / the pressure_reference init step                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "case_fixture, expected_need_reference, expected_reference",
    [
        # Every p_rgh patch is fixedFluxPressure, so no patch fixes the level
        # and setRefCell hands back the case's cell index and value.
        pytest.param("closed", True, {"cell": 2, "value": 50.0, "needs_ref": True}, id="closed"),
        # The totalPressure patch fixes the level, exactly as damBreak's
        # ``atmosphere`` does — setRefCell takes the no-reference path, leaving
        # the cell at -1 and the value at 0, so the case need not carry
        # pRefCell/pRefValue at all.
        pytest.param(
            "open_domain",
            False,
            {"cell": -1, "value": 0.0, "needs_ref": False},
            id="open",
        ),
    ],
)
def test_the_pressure_reference_follows_the_p_rgh_boundary_conditions(
    request: pytest.FixtureRequest,
    case_fixture: str,
    expected_need_reference: bool,
    expected_reference: dict[str, Any],
) -> None:
    run = request.getfixturevalue(case_fixture)
    assert run["need_reference"] is expected_need_reference
    assert run["pressure_reference"] == expected_reference


# --------------------------------------------------------------------------- #
# get_ref_cell_value                                                           #
# --------------------------------------------------------------------------- #


def test_get_ref_cell_value_reads_the_cell_and_is_zero_when_none_is_owned(
    open_domain: Any, closed: Any
) -> None:
    """``0/p_rgh`` is the non-uniform list (100 200 300 400). Sampled on the open
    case because the closed one's start-up levelling moves p_rgh off the file
    values before anything can read them. Where no rank owns the cell,
    OpenFOAM's ``returnReduce(refCelli >= 0 ? field[refCelli] : 0, sumOp)`` is 0.
    """
    assert open_domain["ref_cell_values_of_p_rgh"] == _P_RGH_FILE
    assert closed["ref_cell_value_without_reference_cell"] == 0.0


# --------------------------------------------------------------------------- #
# createFields.H: the same levelling, applied once at start-up                  #
# --------------------------------------------------------------------------- #


def test_startup_levels_p_and_relevels_p_rgh_on_a_closed_domain(closed: Any) -> None:
    """Init already leaves both fields shifted — createFields.H does not wait for
    the pressure corrector, and a frozen-flow case never gets one. Without the
    start-up re-levelling p_rgh keeps whatever level ``0/p_rgh`` carried."""
    assert_allclose(
        closed["p_before"],
        _P_LEVELLED,
        rtol=_RTOL,
        atol=0,
        err_msg="closed: init must apply createFields.H's level shift to p",
    )
    assert_allclose(
        closed["p_rgh_before"],
        _P_RGH_RELEVELLED,
        rtol=_RTOL,
        atol=0,
        err_msg="closed: init must relevel p_rgh from the shifted p",
    )


def test_startup_leaves_an_open_domain_unlevelled(open_domain: Any) -> None:
    """The shift is guarded by needReference(), so an open domain keeps the file
    values — its level is already fixed by the totalPressure patch."""
    assert open_domain["p_rgh_before"] == _P_RGH_FILE


# --------------------------------------------------------------------------- #
# update_absolute_pressure: the pEqn.H tail                                    #
# --------------------------------------------------------------------------- #


def test_the_case_builds_the_hand_derived_density_and_gravity_head(
    closed: Any,
) -> None:
    """Anchors every other expectation in this module: rho and gh are what the
    hand derivation in the docstring assumes."""
    assert_allclose(closed["rho"], _RHO, rtol=_RTOL, atol=0, err_msg="closed")
    assert_allclose(closed["gh"], _GH, rtol=_RTOL, atol=0, err_msg="closed")


def test_the_reference_correction_only_reconstructs_p_on_an_open_domain(
    open_domain: Any,
) -> None:
    """No level shift: p is exactly the ``p_rgh + rho*gh`` reconstruction the
    init step already built, and p_rgh is bit-for-bit what ``0/p_rgh`` holds."""
    assert_allclose(
        open_domain["p_after"],
        _P_RECONSTRUCTED,
        rtol=_RTOL,
        atol=0,
        err_msg="open: p must be p_rgh + rho*gh",
    )
    assert open_domain["p_after"] == open_domain["p_before"]
    assert open_domain["p_rgh_after"] == _P_RGH_FILE


def test_the_level_shift_pins_the_reference_cell_and_relevels_p_rgh(
    closed: Any,
) -> None:
    """The shift is built from the reference cell's own value, so the cell lands
    on pRefValue exactly; ``p_rgh = p - rho*gh`` after it, so the two stay
    consistent."""
    assert closed["ref_cell_value_of_p_after"] == 50.0
    assert_allclose(
        closed["p_after"],
        _P_LEVELLED,
        rtol=_RTOL,
        atol=0,
        err_msg="closed: p shifted by pRefValue - p[pRefCell]",
    )
    assert_allclose(
        closed["p_rgh_after"],
        _P_RGH_RELEVELLED,
        rtol=_RTOL,
        atol=0,
        err_msg="closed: p_rgh relevelled from the shifted p",
    )


# --------------------------------------------------------------------------- #
# p_rgh, not p, is the solved variable                                         #
# --------------------------------------------------------------------------- #


def test_pimple_assembles_and_solves_the_pressure_equation_on_p_rgh() -> None:
    """``fvMatrix::solve`` looks the field name up in ``solvers``; declaring
    p_rgh (and no p) is what makes p_rgh the solved variable. ``pcorr`` is the
    start-up flux projection's own field, not a pressure alias. The equation is
    assembled on it too — ``fvm::laplacian(rAUf, p_rgh)``, never on p."""
    solvers = PimpleFvSolution.model_fields["solvers"].annotation.model_fields
    assert sorted(solvers) == ["U", "UFinal", "p_rgh", "p_rghFinal", "pcorr", "pcorrFinal"]
    laplacians = PimpleFvSchemes.model_fields["laplacianSchemes"].annotation
    assert sorted(f.alias for f in laplacians.model_fields.values()) == [
        "default",
        "laplacian(nuEff,U)",
        "laplacian(rAUf,p_rgh)",
    ]
