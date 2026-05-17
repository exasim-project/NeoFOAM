# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import pytest

from neofoam.framework.initialization.execution.validation import (
    InitializationGraphError,
    validate,
)
from neofoam.framework.initialization.init_step import InitStep


@pytest.mark.parametrize(
    "inits,expected_valid,expected_code,expected_msg_part",
    [
        (
            [
                InitStep("A", depends_on=["missing"], initializer=lambda _ctx: "a"),
                InitStep(
                    "B", depends_on=["also_missing"], initializer=lambda _ctx: "b"
                ),
            ],
            False,
            "missing_dependency",
            "depends on",
        ),
        (
            [
                InitStep("A", depends_on=["B"], initializer=lambda _ctx: "a"),
                InitStep("B", depends_on=["A"], initializer=lambda _ctx: "b"),
            ],
            False,
            "cycle",
            "Circular dependency",
        ),
        (
            [
                InitStep("dup", depends_on=[], initializer=lambda _ctx: 1),
                InitStep("dup", depends_on=[], initializer=lambda _ctx: 2),
            ],
            False,
            "duplicate_name",
            "Duplicate InitStep name",
        ),
        (
            [
                InitStep("A", depends_on=[], initializer=lambda _ctx: "a"),
                InitStep("B", depends_on=["A"], initializer=lambda _ctx: "b"),
            ],
            True,
            None,
            None,
        ),
    ],
    ids=["missing", "cycle", "duplicate", "clean"],
)
def test_validate_graph(inits, expected_valid, expected_code, expected_msg_part):
    report = validate(inits)
    assert report.is_valid is expected_valid
    if expected_valid:
        assert report.diagnostics == ()
        return

    assert len(report.diagnostics) >= 1
    first = report.diagnostics[0]
    assert first.code == expected_code
    assert expected_msg_part in first.message


def test_initialization_graph_error_carries_report():
    inits = [InitStep("A", depends_on=["missing"], initializer=lambda _ctx: "a")]

    report = validate(inits)
    err = InitializationGraphError(report)

    assert err.report is report
    assert "depends on 'missing'" in str(err)
