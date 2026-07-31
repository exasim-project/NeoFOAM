# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The closed-domain pressure reference across MPI ranks.

``pressure_reference.py`` is written for parallel operation — ``need_reference``
OR-reduces the ``pRefCell >= 0`` sentinel, ``get_ref_cell_value`` picks the
owning rank's value with a max-reduction over a one-element/empty field — but
until this module nothing ever ran it on more than one rank, and neither
reduction does anything observable in serial. ``test_pressure_reference.py``
pins the serial behaviour; this is the same claim set on two ranks.

**Why it matters.** ``Foam::setRefCell`` (findRefCell.C, v2406) assigns a
non-negative cell index on *exactly one* rank — with ``pRefPoint``, the rank
whose subdomain contains the point; every other rank gets ``refCelli = -1``,
the same value it uses to mean "no reference needed". Read naively, every rank
but the owner concludes the domain is open, skips the ``pEqn.H`` level shift,
and the run silently develops a per-rank pressure level. The OR-reduction is
what stops that, and the reduction inside ``get_ref_cell_value`` is what lets a
non-owning rank apply the *same* shift.

**The case.** ``cases/vofRow4ClosedRefPoint`` — the closed 4-cell row of
``cases/vofRow4Closed`` with the reference given as a *point* rather than a
cell index (``setRefCell`` reads ``pRefCell`` as a local index on the master
only, which cannot name global cell 2 once the row is halved), and with a
checked-in ``system/decomposeParDict`` that puts the reference cell on rank 1 —
the non-master, the harder half of both reductions. The physical set-up is
otherwise unchanged, so the hand-derived expectations of the serial module carry
over verbatim (see its docstring for the derivation):

    alpha.water = (0, 0.25, 0.75, 1),  rho = (1, 250.75, 750.25, 1000)
    gh = -4.905 everywhere,  p_rgh (from 0/p_rgh) = (100, 200, 300, 400)
    p = p_rgh + rho*gh      = (95.095, -1029.92875, -3379.97625, -4505)
    pRefValue - p[2] = 50 + 3379.97625 shifts p and re-levels p_rgh to
    p     = (3525.07125, 2400.0475, 50, -1075.02375)
    p_rgh = (3529.97625, 3629.97625, 3729.97625, 3829.97625)

Init applies that shift itself (createFields.H), so the worker's ``*_before``
dumps already carry it and the further corrector-tail call it makes is idempotent.

Both ranks run one ``mpirun`` invocation of
``_parallel_pressure_reference_worker.py`` (one ``Foam::Time`` per process) and
each dumps its own JSON; the per-rank slices are put back in global order by
cell-centre x rather than by trusting decomposePar's numbering.

Tolerance: as in the serial module, ``rtol = 1e-13`` (~50 ulp at the O(5e3)
magnitude of the ``rho*gh`` products) for the derived fields, and exact equality
for the two claims that must hold bit-for-bit — the reference cell landing on
``pRefValue``, and the reduced reference value being the *same* number on every
rank.
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

from ...parallel_helpers import NPROCS, decompose, run_mpi

_HERE = Path(__file__).parent
_CASE = _HERE / "cases" / "vofRow4ClosedRefPoint"
_WORKER = _HERE / "_parallel_pressure_reference_worker.py"

# Hand-derived from the case files (see module docstring); identical to the
# serial module's, because the case is the same one with a parallel-safe
# spelling of the same reference cell.
_RHO = [1.0, 250.75, 750.25, 1000.0]
_GH = [-4.905] * 4
_P_LEVELLED = [3525.07125, 2400.0475, 50.0, -1075.02375]
_P_RGH_RELEVELLED = [3529.97625, 3629.97625, 3729.97625, 3829.97625]
# p_rgh in the reference cell as the worker samples it — after init's own
# start-up levelling, so the re-levelled value and not the ``0/p_rgh`` 300.
_P_RGH_AT_REFERENCE_CELL = 3729.97625

_RTOL = 1e-13

# The decomposition the case ships: rank 0 owns global cells 0-1, rank 1 owns
# 2-3, so the reference point (centre of global cell 2) is rank 1's local cell 0.
_EXPECTED_REF_CELLS = [-1, 0]


@pytest.fixture(scope="module")
def ranks(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """Mesh + decompose the case, run the worker on two ranks, return both dumps."""
    case = tmp_path_factory.mktemp("vofRow4ClosedRefPoint") / "case"
    shutil.copytree(_CASE, case)
    subprocess.run(
        ["blockMesh", "-case", str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    # The case ships its own system/decomposeParDict — the split is part of what
    # this module asserts on — so nothing is installed over it here.
    decompose(case)
    run_mpi([sys.executable, str(_WORKER), str(case)], case)
    return [
        json.loads((case / f"parallel_pressure_reference_{rank}.json").read_text())
        for rank in range(NPROCS)
    ]


def _in_global_order(ranks: list[dict[str, Any]], key: str) -> list[float]:
    """Concatenate a per-cell quantity from every rank, ordered by cell-centre x."""
    cells = [(x, value) for rank in ranks for x, value in zip(rank["cell_centres_x"], rank[key])]
    return [value for _x, value in sorted(cells)]


# --------------------------------------------------------------------------- #
# The run really was decomposed                                                #
# --------------------------------------------------------------------------- #


def test_every_rank_reports_the_shared_parallel_session(
    ranks: list[dict[str, Any]],
) -> None:
    """Anchors the rest: two ranks of one MPI run, not two serial runs."""
    assert [rank["rank"] for rank in ranks] == [0, 1]
    assert [rank["nProcs"] for rank in ranks] == [NPROCS, NPROCS]
    assert all(rank["parRun"] for rank in ranks)


def test_the_reference_cell_is_owned_by_the_non_master_rank(
    ranks: list[dict[str, Any]],
) -> None:
    """setRefCell finds ``pRefPoint`` on rank 1 only; rank 0 gets the same -1 it
    would get on an open domain. Everything below is about that asymmetry."""
    assert [rank["pressure_reference"]["pRefCell"] for rank in ranks] == (_EXPECTED_REF_CELLS)


# --------------------------------------------------------------------------- #
# need_reference: the OR-reduction                                             #
# --------------------------------------------------------------------------- #


def test_need_reference_is_true_on_every_rank_of_a_closed_domain(
    ranks: list[dict[str, Any]],
) -> None:
    """Rank 0 holds ``pRefCell = -1`` and must still answer True — without the
    OR-reduction it would read its own sentinel as "open domain" and skip the
    level shift the other rank applies."""
    assert [rank["need_reference"] for rank in ranks] == [True, True]


def test_the_cached_pressure_reference_flag_agrees_on_every_rank(
    ranks: list[dict[str, Any]],
) -> None:
    """The flag the init step stores in ``ctx.models["pressure_reference"]`` is
    what the solver actually branches on every pressure corrector."""
    assert [rank["pressure_reference"]["needsRef"] for rank in ranks] == [True, True]
    assert [rank["pressure_reference"]["pRefValue"] for rank in ranks] == [50.0, 50.0]


# --------------------------------------------------------------------------- #
# get_ref_cell_value: the max-reduction over one owner                         #
# --------------------------------------------------------------------------- #


def test_get_ref_cell_value_carries_the_owning_ranks_value_to_every_rank(
    ranks: list[dict[str, Any]],
) -> None:
    """Rank 0 does not hold the cell at all, so the number it reports can only
    have come over MPI — and it must be bit-for-bit rank 1's."""
    assert [rank["ref_cell_value_of_p_rgh"] for rank in ranks] == (
        [_P_RGH_AT_REFERENCE_CELL] * NPROCS
    )


def test_get_ref_cell_value_is_zero_when_no_rank_owns_a_reference_cell(
    ranks: list[dict[str, Any]],
) -> None:
    """Every rank passing -1 is OpenFOAM's ``returnReduce(0, sumOp)`` case; the
    max-over-empty-fields composition must answer 0 and not ``-VGREAT``."""
    assert [rank["ref_cell_value_without_reference_cell"] for rank in ranks] == ([0.0] * NPROCS)


# --------------------------------------------------------------------------- #
# update_absolute_pressure: one pressure level over the whole domain           #
# --------------------------------------------------------------------------- #


def test_the_decomposed_case_builds_the_hand_derived_density_and_gravity_head(
    ranks: list[dict[str, Any]],
) -> None:
    """Anchors every expectation below: decomposing the case did not change the
    fields the level shift is computed from."""
    assert_allclose(
        _in_global_order(ranks, "rho"),
        _RHO,
        rtol=_RTOL,
        atol=0,
        err_msg="vofRow4ClosedRefPoint on 2 ranks",
    )
    assert_allclose(
        _in_global_order(ranks, "gh"),
        _GH,
        rtol=_RTOL,
        atol=0,
        err_msg="vofRow4ClosedRefPoint on 2 ranks",
    )


def test_startup_levels_the_pressure_across_ranks(ranks: list[dict[str, Any]]) -> None:
    """``p_before`` is what *init* leaves behind: createFields.H's shift, already
    applied on both ranks — including the one that owns no reference cell."""
    assert_allclose(
        _in_global_order(ranks, "p_before"),
        _P_LEVELLED,
        rtol=_RTOL,
        atol=0,
        err_msg="vofRow4ClosedRefPoint on 2 ranks: init must level p by pRefValue - p[2]",
    )


def test_absolute_pressure_is_level_shifted_across_ranks(
    ranks: list[dict[str, Any]],
) -> None:
    """One shift, applied on both ranks, giving the serial answer: the level is
    a property of the domain and not of the subdomain."""
    assert_allclose(
        _in_global_order(ranks, "p_after"),
        _P_LEVELLED,
        rtol=_RTOL,
        atol=0,
        err_msg="vofRow4ClosedRefPoint on 2 ranks: p shifted by pRefValue - p[2]",
    )


def test_p_rgh_is_relevelled_across_ranks(ranks: list[dict[str, Any]]) -> None:
    assert_allclose(
        _in_global_order(ranks, "p_rgh_after"),
        _P_RGH_RELEVELLED,
        rtol=_RTOL,
        atol=0,
        err_msg="vofRow4ClosedRefPoint on 2 ranks: p_rgh relevelled from shifted p",
    )


def test_the_reference_cell_lands_on_the_reference_value_for_every_rank(
    ranks: list[dict[str, Any]],
) -> None:
    """The shift is built from the reference cell's own value, so after it the
    cell reads exactly ``pRefValue`` — and every rank can say so."""
    assert [rank["ref_cell_value_of_p_after"] for rank in ranks] == [50.0] * NPROCS
