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

import json
from pathlib import Path
from typing import Any

import pytest

from neofoam.tooling.verification.execute import (
    CASE_SETUP_FAILED,
    NATIVE_FAILED,
    POSTPROCESS_NOT_IMPLEMENTED,
    SOLVER_FAILED,
    UNSUPPORTED_CASE,
)
from neofoam.tooling.verification.runner import _case_dir, _compare, _decide, _run
from neofoam.tooling.verification.study import Case, Study

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


def test_decide_reports_unsupported_case_when_postprocess_not_implemented() -> None:
    """A candidate whose Allrun invoked ``-postProcess`` is a harness coverage
    gap, not a solver crash — UNSUPPORTED_CASE, not SOLVER_FAILED."""
    neo = {
        "finished": False,
        "timed_out": False,
        "reason": POSTPROCESS_NOT_IMPLEMENTED,
    }
    outcome, detail, diffs = _decide(_FINISHED, neo, Path("n"), Path("m"), ["U"])
    assert outcome == UNSUPPORTED_CASE
    assert detail == POSTPROCESS_NOT_IMPLEMENTED
    assert diffs == []


def test_decide_still_reports_solver_failed_for_other_reasons() -> None:
    neo = {"finished": False, "timed_out": False, "reason": "matrix is singular"}
    outcome, _, _ = _decide(_FINISHED, neo, Path("n"), Path("m"), ["U"])
    assert outcome == SOLVER_FAILED


_PATCH = {"system/fvSolution": {"advectionScheme": "isoAdvector"}}


def _study_with_patch() -> Study:
    return Study(
        title="vof",
        config_path=Path("/studies/config-isoadvector.yaml"),
        cases=[],
        tier_titles={},
        config={"neo_patch": _PATCH},
        apps=("neofoam solver incompressiblevof",),
    )


def _vof_case() -> Case:
    return Case(
        id="interIsoFoam__damBreak",
        name="interIsoFoam/damBreak",
        path=Path("/tut/interIsoFoam/damBreak"),
        native_solver="interIsoFoam",
        app="neofoam solver incompressiblevof",
        fields=("alpha.water",),
    )


def _write_status(work: Path, case_id: str, solver: str, status: dict) -> None:
    (work / case_id).mkdir(parents=True, exist_ok=True)
    (work / case_id / f"{solver}.status.json").write_text(json.dumps(status))


def test_compare_records_the_study_identity_in_the_result(tmp_path: Path) -> None:
    """Each result names the config + neo_patch it came from, so a results dir mixing
    two sweeps (strict vs an isoAdvector neo_patch) cannot masquerade as one."""
    study, case = _study_with_patch(), _vof_case()
    work = tmp_path / "work"
    # Legacy combined statuses (authoritative) so no OpenFOAM read is needed: native
    # finished, candidate failed → SOLVER_FAILED, no field compare.
    _write_status(work, case.id, "interIsoFoam", {"finished": True, "seconds": 1.0})
    _write_status(
        work,
        case.id,
        "incompressiblevof",
        {"finished": False, "timed_out": False, "reason": "boom", "seconds": 2.0},
    )
    out = tmp_path / "result.json"

    _compare(study, case, work, out)

    record = json.loads(out.read_text())
    assert record["study_config"] == "config-isoadvector.yaml"
    assert record["neo_patch"] == _PATCH


def test_run_falls_back_to_first_token_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The combined ``run`` (packaged driver / VoF study) writes its own
    ``.status.json`` straight from ``run_allrun``'s primary-log check
    (``_status_from_rundir`` then trusts that file verbatim, see
    ``test_run_split.py::test_status_from_rundir_prefers_a_combined_status_json``)
    — so unlike the split pipeline, ``_run`` itself must resolve the real
    ``log.<first token>`` when ``application="…"; runApplication ${application}``
    (unquoted) re-splits the multi-word neofoam command.
    """
    import neofoam.tooling.verification.runner as runner_mod

    study, case = _study_with_patch(), _vof_case()
    work = tmp_path / "work"

    def fake_stage(*args: Any, **kwargs: Any) -> None:
        _case_dir(work, case, "incompressiblevof").mkdir(parents=True, exist_ok=True)

    def fake_run_allrun(
        case_dir: Path, solver_log: str, timeout: int = 1800
    ) -> dict[str, object]:
        # Only the first-token fallback log was written, as the real
        # `runApplication ${application}` (unquoted) idiom does — the primary
        # `log.<full command>` this call was asked to check never exists.
        (case_dir / "log.neofoam").write_text("Starting time loop\nTime = 1\nEnd\n")
        return {
            "finished": False,
            "timed_out": False,
            "reason": "no solver log written",
            "seconds": 1.0,
        }

    monkeypatch.setattr(runner_mod, "stage", fake_stage)
    monkeypatch.setattr(runner_mod, "run_allrun", fake_run_allrun)

    status_out = tmp_path / "status.json"
    _run(study, case, "incompressiblevof", work, status_out)

    status = json.loads(status_out.read_text())
    assert status["finished"] is True
    assert status["reason"] == ""
