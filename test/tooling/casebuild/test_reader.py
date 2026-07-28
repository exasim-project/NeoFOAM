# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the subprocess-isolated field reader (``CaseDir.read_field``).

Proves the reader returns a finite internal field and stays consistent across two
reads in one interpreter — the subprocess-per-read isolation that dodges the
``Foam::Time`` double-construction ``nan`` corruption. The committed ``cavity``
template is staged into ``tmp_path`` and meshed with ``block_mesh`` first.

Two cases carry what a *written* case can hold that the reader must survive, both
taken from the drop-in verification sweep:

* ``fanCurveBoundary`` — a ``fanPressure`` boundary condition whose fan curve is a
  case-relative ``tableFile``. Constructing that boundary condition opens the table
  against the working directory, which is not the case, so it can only be read by a
  reader that never constructs boundary conditions.
* ``includeFuncControl`` — a ``#includeFunc`` of a case-local ``system/`` file, which
  resolves only when ``$FOAM_CASE`` points at the case being read. Its mesh is
  committed rather than generated, because ``block_mesh`` would read the same
  controlDict in *this* interpreter, where an OpenFOAM fatal cannot be caught.

Both fields carry a pre-written non-uniform ``internalField`` so the assertions are
on values the reader had to read, not on a shape it could have invented.
"""

from pathlib import Path

import numpy as np

from neofoam.tooling.casebuild import block_mesh, from_template

CASES = Path(__file__).parent / "cases"
CAVITY = CASES / "cavity"
FAN_CURVE_BOUNDARY = CASES / "fanCurveBoundary"
INCLUDE_FUNC_CONTROL = CASES / "includeFuncControl"


def test_read_field_returns_finite_internal_field(tmp_path: Path) -> None:
    case = (from_template(CAVITY) | block_mesh()).build_at(tmp_path / "c")
    p = case.read_field("p")  # initial p (from 0.orig) at the latest (only) time
    assert p.ndim == 1 and p.size > 0  # scalar field -> (N,)
    assert np.all(np.isfinite(p))  # subprocess isolation -> no Foam::Time nan corruption


def test_read_field_twice_in_one_process_stays_consistent(tmp_path: Path) -> None:
    # Two Foam::Time constructions in ONE interpreter corrupt global state (later
    # reads return nan); read_field spawns a fresh process per read, so both agree.
    case = (from_template(CAVITY) | block_mesh()).build_at(tmp_path / "c")
    first = case.read_field("p")
    second = case.read_field("p")
    assert np.all(np.isfinite(second))
    np.testing.assert_array_equal(first, second)


def test_read_field_reads_field_with_unconstructable_boundary_condition(tmp_path: Path) -> None:
    case = (from_template(FAN_CURVE_BOUNDARY) | block_mesh()).build_at(tmp_path / "c")
    p = case.read_field("p")
    # Exact: these are the nine numbers written in 0.orig/p, not a computed result.
    np.testing.assert_array_equal(p, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])


def test_read_field_resolves_case_local_include_directive(tmp_path: Path) -> None:
    case = from_template(INCLUDE_FUNC_CONTROL).build_at(tmp_path / "c")  # mesh is committed
    p = case.read_field("p")
    # Exact: these are the nine numbers written in 0/p, not a computed result.
    np.testing.assert_array_equal(p, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
