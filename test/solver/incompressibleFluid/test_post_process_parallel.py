# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end: a decomposed run writes the same CSVs as the serial one.

**What this proves.** Every rank evaluates every table over its own cells, the
aggregators reduce their per-bin numbers over the ranks
(``neofoam.postprocess._reduce``, one ``gSum``/``gMax``/``gMin`` per element) and
only the master rank writes — so ``mpirun -np 2`` on the decomposed cavity has to
produce, file for file and row for row, what one process produces. That is the
whole contract; the unit tests can only pin the delegation, because a collective
needs real ranks.

**The tables.** ``cases/postprocess_parallel`` declares one per shape whose
reduction could go wrong on this split: a plain ``sum`` (the mesh volume, an
exact literal), ``max`` and ``min`` (whose empty bins are filled with the
``±GREAT`` sentinel locally, so a missing reduction is loud), a masked *and*
binned ``volIntegrate`` (the R6 row-count check: three ``directional`` bins from
the binner's spec, not from the local data), a ``surfIntegrate`` over a patch
both ranks own, and a ``plane | sample | mean`` whose surface lies entirely in
rank 0's subdomain, so rank 1 enters the collectives with an empty selection.

**Why the tolerance is reachable at all.** Serial and decomposed PCG take
different iteration paths, and at the preset's ``relTol 0.05`` on the first
pressure corrector the two solutions part company at ~1e-5 — a table comparison
would then measure the linear solver, not the reduction. ``_EXACT_SOLVES``
tightens every solve to ``1e-14`` absolute so both runs reach the same discrete
solution; the residual difference is then the summation order alone. Measured
worst case is 1.6e-12 (on the near-cancelling middle bin of ``binned_p``), two
decades inside ``_RTOL``.

**Not comparable to the serial module.** ``test_post_process.py`` runs the
preset untouched, so its numbers are not these.

**Skip.** ``mpirun``/``decomposePar`` are skipped when absent, the convention the
other parallel test in this package (``test_hotRoom_telemetry.py``) already uses.
``FOAM_SIGFPE`` is disabled inside ``_parallel_driver.py``, as in every solver
run here.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neofoam.tooling.casebuild import CaseDir, patch

from .solved_case import built_cavity, solve_serially

_HERE = Path(__file__).parent
_CASES = _HERE / "cases"
_REPO_ROOT = _HERE.parent.parent.parent

#: Shared with the other parallel tests: ``simple (2 1 1)``, two subdomains.
_DECOMPOSE_PAR_DICT = _HERE / "_parallel_decomposeParDict"
_PARALLEL_DRIVER = _HERE / "_parallel_driver.py"

#: See the module docstring: without this the linear solver, not the reduction,
#: sets the difference between the two runs.
_EXACT_SOLVES = patch(
    "system/fvSolution",
    **{
        "solvers.p.tolerance": 1e-14,
        "solvers.p.relTol": 0.0,
        "solvers.pFinal.tolerance": 1e-14,
        "solvers.pFinal.relTol": 0.0,
        "solvers.U.tolerance": 1e-14,
        "solvers.UFinal.tolerance": 1e-14,
    },
)

#: Summation order alone, two decades of headroom over the measured 1.6e-12.
_RTOL = 1e-10

#: Every table ``cases/postprocess_parallel`` declares.
TABLES = ["mesh_volume", "p_max", "p_min", "binned_p", "wall_p", "plane_speed"]


def _mpi_available() -> bool:
    return shutil.which("mpirun") is not None and shutil.which("decomposePar") is not None


def _cavity_with_the_parallel_tables(dest: Path) -> CaseDir:
    """The meshed cavity carrying the declaration, its solves tightened, unsolved."""
    case = built_cavity(dest, declarations_dir=_CASES / "postprocess_parallel")
    _EXACT_SOLVES(case)
    return case


def _run_decomposed(case: CaseDir) -> None:
    """``decomposePar`` then ``mpirun -np 2`` from the case root, as ``Allrun`` does."""
    shutil.copyfile(_DECOMPOSE_PAR_DICT, case.path / "system" / "decomposeParDict")
    decomposed = subprocess.run(
        ["decomposePar", "-case", str(case.path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert decomposed.returncode == 0, f"decomposePar failed:\n{decomposed.stderr[-2000:]}"

    solved = subprocess.run(
        ["mpirun", "-np", "2", sys.executable, str(_PARALLEL_DRIVER)],
        cwd=case.path,
        capture_output=True,
        text=True,
        # Bounded: a rank-local reduction shows up as a deadlock (one rank
        # inside a collective the other never enters), which would otherwise
        # hang the suite.
        timeout=300,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
    )
    assert solved.returncode == 0, (
        f"the decomposed run failed:\nstdout:\n{solved.stdout[-4000:]}\n"
        f"stderr:\n{solved.stderr[-4000:]}"
    )


def _table(case: CaseDir, name: str) -> tuple[list[str], "np.ndarray[Any, Any]"]:
    """``postProcessing/<name>.csv`` as its header row and its numbers."""
    with (case.path / "postProcessing" / f"{name}.csv").open(newline="") as handle:
        header, *rows = csv.reader(handle)
    return header, np.array([[float(value) for value in row] for row in rows])


@pytest.fixture(scope="module")
def serial_cavity(tmp_path_factory: pytest.TempPathFactory) -> CaseDir:
    """The reference: one process, the same case files."""
    case = _cavity_with_the_parallel_tables(tmp_path_factory.mktemp("serial") / "cavity")
    solve_serially(case)
    return case


@pytest.fixture(scope="module")
def decomposed_cavity(tmp_path_factory: pytest.TempPathFactory) -> CaseDir:
    """The subject: the same case cut in two along x and run under ``mpirun``."""
    case = _cavity_with_the_parallel_tables(tmp_path_factory.mktemp("parallel") / "cavity")
    _run_decomposed(case)
    return case


@pytest.mark.skipif(not _mpi_available(), reason="mpirun/decomposePar not available")
@pytest.mark.parametrize("name", TABLES)
def test_a_decomposed_table_equals_the_serial_one(
    name: str, serial_cavity: CaseDir, decomposed_cavity: CaseDir
) -> None:
    expected_header, expected = _table(serial_cavity, name)

    header, actual = _table(decomposed_cavity, name)

    assert header == expected_header
    assert actual.shape == expected.shape
    assert_allclose(actual, expected, rtol=_RTOL, err_msg=f"{name}.csv, cavity3x3 on two ranks")


@pytest.mark.skipif(not _mpi_available(), reason="mpirun/decomposePar not available")
def test_a_binned_table_keeps_the_bins_its_binner_declared_on_every_rank(
    decomposed_cavity: CaseDir,
) -> None:
    # R6: a row count taken from the local data would differ between the ranks.
    header, rows = _table(decomposed_cavity, "binned_p")

    assert header == ["time", "bin", "binned_p"]
    assert [row[1] for row in rows] == [0.0, 1.0, 2.0] * 3


@pytest.mark.skipif(not _mpi_available(), reason="mpirun/decomposePar not available")
def test_only_the_master_rank_writes_and_it_writes_to_the_case_root(
    decomposed_cavity: CaseDir,
) -> None:
    written = decomposed_cavity.path.rglob("postProcessing")

    assert [path.relative_to(decomposed_cavity.path).as_posix() for path in written] == [
        "postProcessing"
    ]
    assert sorted(path.stem for path in (decomposed_cavity.path / "postProcessing").iterdir()) == (
        sorted(TABLES)
    )
