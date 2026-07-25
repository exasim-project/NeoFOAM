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

from neofoam.tooling.workflow.study.cases import Case, Study
from neofoam.tooling.workflow.study.execute import (
    CASE_SETUP_FAILED,
    NATIVE_FAILED,
    POSTPROCESS_NOT_IMPLEMENTED,
    SOLVER_FAILED,
    UNSUPPORTED_CASE,
)
from neofoam.tooling.workflow.study.runner import _case_dir, _compare, _decide

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
    neo = {"finished": False, "reason": POSTPROCESS_NOT_IMPLEMENTED}
    outcome, detail, diffs = _decide(_FINISHED, neo, Path("n"), Path("m"), ["U"])
    assert outcome == UNSUPPORTED_CASE
    assert detail == POSTPROCESS_NOT_IMPLEMENTED
    assert diffs == []


def test_decide_still_reports_solver_failed_for_other_reasons() -> None:
    neo = {"finished": False, "reason": "matrix is singular"}
    outcome, _, _ = _decide(_FINISHED, neo, Path("n"), Path("m"), ["U"])
    assert outcome == SOLVER_FAILED


_PATCH = {"system/fvSolution": {"advectionScheme": "isoAdvector"}}


def _study_with_patch() -> Study:
    return Study(
        title="vof",
        config_path=Path("/studies/config-isoadvector.yaml"),
        cases=[],
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


def _write_run_dir(cases_root: Path, case: Case, solver: str, log: str) -> None:
    """A finished run dir as the shell ``run`` rule leaves it: swap stamp + solver log.

    ``_status_from_rundir`` reads exactly these two, so a ``log`` ending in ``End``
    reads finished and anything else reads failed — no OpenFOAM needed.
    """
    run_dir = _case_dir(cases_root, case, solver)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".swapped.json").write_text(json.dumps({"solver": solver}))
    (run_dir / ".seconds").write_text("1.0")
    log_name = f"log.{case.native_solver}" if solver == case.native_label else f"log.{case.app}"
    (run_dir / log_name).write_text(log)


def test_compare_records_the_study_identity_in_the_result(tmp_path: Path) -> None:
    """Each result names the config + neo_patch it came from, so a results dir mixing
    two sweeps (strict vs an isoAdvector neo_patch) cannot masquerade as one."""
    study, case = _study_with_patch(), _vof_case()
    cases_root = tmp_path / "cases"
    # Native finished, candidate did not → SOLVER_FAILED, so no field compare runs
    # and the assertions below need no staged case.
    _write_run_dir(cases_root, case, "interIsoFoam", "Time = 1\nEnd\n")
    _write_run_dir(cases_root, case, "incompressiblevof", "Time = 1\n--> FOAM FATAL ERROR: boom\n")
    out = tmp_path / "result.json"

    _compare(study, case, cases_root, out)

    record = json.loads(out.read_text())
    assert record["study_config"] == "config-isoadvector.yaml"
    assert record["neo_patch"] == _PATCH
    assert [c["outcome"] for c in record["candidates"]] == [SOLVER_FAILED]
