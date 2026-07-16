# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the subprocess-isolated field reader (``CaseDir.read_field``).

Proves the reader returns a finite internal field and stays consistent across two
reads in one interpreter — the subprocess-per-read isolation that dodges the
``Foam::Time`` double-construction ``nan`` corruption. The committed ``cavity``
template is staged into ``tmp_path`` and meshed with ``block_mesh`` first.
"""

from pathlib import Path

import numpy as np

from neofoam.tooling.casebuild import block_mesh, from_template

CAVITY = Path(__file__).parent / "cases" / "cavity"


def test_read_field_returns_finite_internal_field(tmp_path: Path) -> None:
    case = (from_template(CAVITY) | block_mesh()).build_at(tmp_path / "c")
    p = case.read_field("p")  # initial p (from 0.orig) at the latest (only) time
    assert p.ndim == 1 and p.size > 0  # scalar field -> (N,)
    assert np.all(
        np.isfinite(p)
    )  # subprocess isolation -> no Foam::Time nan corruption


def test_read_field_twice_in_one_process_stays_consistent(tmp_path: Path) -> None:
    # Two Foam::Time constructions in ONE interpreter corrupt global state (later
    # reads return nan); read_field spawns a fresh process per read, so both agree.
    case = (from_template(CAVITY) | block_mesh()).build_at(tmp_path / "c")
    first = case.read_field("p")
    second = case.read_field("p")
    assert np.all(np.isfinite(second))
    np.testing.assert_array_equal(first, second)
