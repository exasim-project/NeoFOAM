# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the field-comparison helpers.

The helpers read each case in an isolated subprocess (OpenFOAM's per-process
global state corrupts a second in-process ``Foam::Time`` read), so two reads
must reflect their own cases and identical cases must compare equal.
"""

import shutil
from pathlib import Path

from .comparison_helpers import (
    compare_fields_numerically,
    read_internal_fields,
    setup_case,
)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_PITZ = _REPO_ROOT / "tutorials" / "pitzDaily"


def test_identical_cases_compare_equal(tmp_path: Path) -> None:
    case_a = tmp_path / "a"
    setup_case(_PITZ, case_a, end_time=0.001, write_interval=0.001)
    case_b = tmp_path / "b"
    shutil.copytree(case_a, case_b)

    fields_a = read_internal_fields(case_a, case_a / "0", ["p", "U"])
    fields_b = read_internal_fields(case_b, case_b / "0", ["p", "U"])

    for name in ("p", "U"):
        match, max_abs, _ = compare_fields_numerically(
            fields_a[name], fields_b[name], name
        )
        assert match is True
        assert max_abs == 0.0
