# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A staging failure becomes an outcome, never an aborted DAG.

Some tutorials ship a ``controlDict`` casebuild cannot parse (an unresolvable
``#includeFunc``); staging them raises. The rule that runs a case must still exit
cleanly and record the failure as data — otherwise one unparsable case strands
the whole report (``verify_report`` needs every case's result). This pins the
classifier that turns a staging failure into ``CASE_SETUP_FAILED``, ahead of the
native/solver verdicts so it is never mistaken for either.
"""

from pathlib import Path

from neofoam.tooling.verification.execute import (
    CASE_SETUP_FAILED,
    NATIVE_FAILED,
)
from neofoam.tooling.verification.runner import _decide

_FINISHED = {"finished": True}


def test_decide_reports_stage_failure_before_native_failure() -> None:
    native = {
        "stage_failed": True,
        "reason": "staging failed: cannot parse controlDict",
    }
    outcome, detail, diffs = _decide(native, _FINISHED, Path("n"), Path("m"), ["U"])
    assert outcome == CASE_SETUP_FAILED
    assert "cannot parse controlDict" in detail
    assert diffs == []


def test_decide_still_reports_native_failure_when_nothing_staged_wrong() -> None:
    native = {"finished": False, "reason": "native did not finish"}
    outcome, _, _ = _decide(native, _FINISHED, Path("n"), Path("m"), ["U"])
    assert outcome == NATIVE_FAILED
