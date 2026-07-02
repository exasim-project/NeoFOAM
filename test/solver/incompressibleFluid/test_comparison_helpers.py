# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Regression spec for the field-comparison helpers.

The old in-process reader constructed ``Foam::Time`` repeatedly in one
interpreter; OpenFOAM's per-process global state made the second read
silently return the *first* case's fields, so comparing two genuinely
different solutions reported ``max abs = 0``. The helpers must read each
case in an isolated subprocess and detect real differences.
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from .comparison_helpers import (  # noqa: E402
    compare_fields_numerically,
    read_internal_fields,
    setup_case,
)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_PITZ = _REPO_ROOT / "tutorials" / "pitzDaily"


def _perturb_pressure(case: Path) -> None:
    field = case / "0" / "p"
    text = field.read_text()
    assert "uniform 0;" in text
    field.write_text(text.replace("uniform 0;", "uniform 1;", 1))


def test_second_read_in_same_process_sees_the_second_case(tmp_path: Path) -> None:
    """The regression: back-to-back reads of two cases must not alias."""
    case_a = tmp_path / "a"
    setup_case(_PITZ, case_a, end_time=0.001, write_interval=0.001)
    case_b = tmp_path / "b"
    shutil.copytree(case_a, case_b)
    _perturb_pressure(case_b)

    fields_a = read_internal_fields(case_a, case_a / "0", ["p", "U"])
    fields_b = read_internal_fields(case_b, case_b / "0", ["p", "U"])

    assert fields_a["p"].max() == pytest.approx(0.0)
    assert fields_b["p"].min() == pytest.approx(1.0)  # NOT case a's values

    match, max_abs, _ = compare_fields_numerically(fields_a["p"], fields_b["p"], "p")
    assert match is False
    assert max_abs == pytest.approx(1.0)


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
