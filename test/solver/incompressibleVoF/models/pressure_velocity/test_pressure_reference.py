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

* ``cases/vofRow4Closed`` — one ``wall`` patch, ``fixedFluxPressure`` (fixes a
  *gradient*): closed, ``PIMPLE { pRefCell 2; pRefValue 50; }``. Neither is the
  default 0, so honouring the cell index and the value are separate claims.
* ``cases/vofRow4Open`` — the interFoam damBreak ``atmosphere`` arrangement, a
  ``totalPressure`` patch (a ``fixedValue`` descendant): open, and the PIMPLE
  dict deliberately carries no ``pRefCell``/``pRefValue`` at all.

Both are the ``cases/vofRow4`` unit box cut into 4 cells along x (cell centres
at y = 0.5) with the interFoam damBreak properties, so every expected value is
hand-derivable:

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
import shutil
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


def _run_worker(case_name: str, tmp_path_factory: pytest.TempPathFactory) -> Any:
    case = tmp_path_factory.mktemp(case_name) / "case"
    shutil.copytree(_CASES / case_name, case)
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
    return _run_worker("vofRow4Closed", tmp_path_factory)


@pytest.fixture(scope="session")
def open_domain(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return _run_worker("vofRow4Open", tmp_path_factory)


# --------------------------------------------------------------------------- #
# need_reference / the pressure_reference init step                            #
# --------------------------------------------------------------------------- #


def test_need_reference_is_true_for_a_closed_domain(closed: Any) -> None:
    """Every p_rgh patch is fixedFluxPressure, so no patch fixes the level and
    setRefCell hands back the case's cell index."""
    assert closed["need_reference"] is True


def test_need_reference_is_false_for_an_open_domain(open_domain: Any) -> None:
    """The totalPressure patch fixes the level, exactly as damBreak's
    ``atmosphere`` does — setRefCell takes the no-reference path and the cell
    index comes back negative."""
    assert open_domain["need_reference"] is False


def test_pressure_reference_carries_the_flag_and_the_cell_on_a_closed_domain(
    closed: Any,
) -> None:
    assert closed["pressure_reference"] == {
        "pRefCell": 2,
        "pRefValue": 50.0,
        "needsRef": True,
    }


def test_pressure_reference_has_no_cell_on_an_open_domain(open_domain: Any) -> None:
    """setRefCell leaves the cell at -1 and the value at 0 when the domain is
    open, so the case need not carry pRefCell/pRefValue at all."""
    assert open_domain["pressure_reference"] == {
        "pRefCell": -1,
        "pRefValue": 0.0,
        "needsRef": False,
    }


# --------------------------------------------------------------------------- #
# get_ref_cell_value                                                           #
# --------------------------------------------------------------------------- #


def test_get_ref_cell_value_returns_the_cells_value(open_domain: Any) -> None:
    """``0/p_rgh`` is the non-uniform list (100 200 300 400). Sampled on the open
    case because the closed one's start-up levelling moves p_rgh off the file
    values before anything can read them."""
    assert open_domain["ref_cell_values_of_p_rgh"] == _P_RGH_FILE


def test_get_ref_cell_value_is_zero_when_no_rank_owns_a_reference_cell(
    closed: Any,
) -> None:
    """OpenFOAM's ``returnReduce(refCelli >= 0 ? field[refCelli] : 0, sumOp)``
    is 0 when the cell is owned nowhere."""
    assert closed["ref_cell_value_without_reference_cell"] == 0.0


# --------------------------------------------------------------------------- #
# createFields.H: the same levelling, applied once at start-up                  #
# --------------------------------------------------------------------------- #


def test_startup_levels_the_absolute_pressure_on_a_closed_domain(closed: Any) -> None:
    """Init already leaves p shifted — createFields.H does not wait for the
    pressure corrector, and a frozen-flow case never gets one."""
    assert_allclose(
        closed["p_before"],
        _P_LEVELLED,
        rtol=_RTOL,
        atol=0,
        err_msg="vofRow4Closed: init must apply createFields.H's level shift to p",
    )


def test_startup_relevels_p_rgh_on_a_closed_domain(closed: Any) -> None:
    """The field the solver writes out: without the start-up re-levelling it
    keeps whatever level ``0/p_rgh`` happened to carry."""
    assert_allclose(
        closed["p_rgh_before"],
        _P_RGH_RELEVELLED,
        rtol=_RTOL,
        atol=0,
        err_msg="vofRow4Closed: init must relevel p_rgh from the shifted p",
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
    assert_allclose(closed["rho"], _RHO, rtol=_RTOL, atol=0, err_msg="vofRow4Closed")
    assert_allclose(closed["gh"], _GH, rtol=_RTOL, atol=0, err_msg="vofRow4Closed")


def test_absolute_pressure_is_reconstructed_from_p_rgh_on_an_open_domain(
    open_domain: Any,
) -> None:
    assert_allclose(
        open_domain["p_after"],
        _P_RECONSTRUCTED,
        rtol=_RTOL,
        atol=0,
        err_msg="vofRow4Open: p must be p_rgh + rho*gh",
    )


def test_the_reference_correction_leaves_an_open_domain_untouched(
    open_domain: Any,
) -> None:
    """No level shift: p is exactly the reconstruction the init step already
    built, and p_rgh is bit-for-bit what ``0/p_rgh`` holds."""
    assert open_domain["p_after"] == open_domain["p_before"]
    assert open_domain["p_rgh_after"] == _P_RGH_FILE


def test_absolute_pressure_is_pinned_to_the_reference_value_on_a_closed_domain(
    closed: Any,
) -> None:
    """The shift is built from the reference cell's own value, so the cell
    lands on pRefValue exactly."""
    assert closed["ref_cell_value_of_p_after"] == 50.0


def test_absolute_pressure_is_level_shifted_on_a_closed_domain(closed: Any) -> None:
    assert_allclose(
        closed["p_after"],
        _P_LEVELLED,
        rtol=_RTOL,
        atol=0,
        err_msg="vofRow4Closed: p shifted by pRefValue - p[pRefCell]",
    )


def test_p_rgh_is_relevelled_from_the_shifted_pressure_on_a_closed_domain(
    closed: Any,
) -> None:
    """``p_rgh = p - rho*gh`` after the shift — the two stay consistent."""
    assert_allclose(
        closed["p_rgh_after"],
        _P_RGH_RELEVELLED,
        rtol=_RTOL,
        atol=0,
        err_msg="vofRow4Closed: p_rgh relevelled from the shifted p",
    )


# --------------------------------------------------------------------------- #
# p_rgh, not p, is the solved variable                                         #
# --------------------------------------------------------------------------- #


def test_pimple_declares_a_solver_for_p_rgh_and_none_for_p() -> None:
    """``fvMatrix::solve`` looks the field name up in ``solvers``; declaring
    p_rgh (and no p) is what makes p_rgh the solved variable. ``pcorr`` is the
    start-up flux projection's own field, not a pressure alias."""
    solvers = PimpleFvSolution.model_fields["solvers"].annotation.model_fields
    assert sorted(solvers) == ["U", "UFinal", "p_rgh", "p_rghFinal", "pcorr", "pcorrFinal"]


def test_pimple_declares_the_pressure_laplacian_on_p_rgh() -> None:
    """``fvm::laplacian(rAUf, p_rgh)`` — the pressure equation is assembled on
    p_rgh, never on p."""
    laplacians = PimpleFvSchemes.model_fields["laplacianSchemes"].annotation
    assert sorted(f.alias for f in laplacians.model_fields.values()) == [
        "default",
        "laplacian(nuEff,U)",
        "laplacian(rAUf,p_rgh)",
    ]
