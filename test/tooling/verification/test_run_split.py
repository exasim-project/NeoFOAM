# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""build → swap → run split: swap neutralises, compare interprets from the run dir.

The run rule is plain shell (``./Allrun``) — no timeout, no interpretation. Two
invariants make that safe and correct, and are pinned here without OpenFOAM:

* ``swap`` *neutralises* a case that cannot run (staging crashed, or no unique
  solver token) — its ``Allrun`` becomes a no-op — so the shell run never executes
  an un-swapped native solver in a candidate's dir (which would score a false
  MATCHED), and the reason is carried in the swap stamp.
* ``_status_from_rundir`` reconstructs the outcome at compare time by reading the
  run dir: a terminal swap stamp short-circuits, else the solver log's trailing
  ``End`` decides finished, and the reason comes from the log.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neofoam.tooling.verification import runner
from neofoam.tooling.verification.runner import _build, _status_from_rundir, _swap
from neofoam.tooling.verification.stage import NoSwapPoint
from neofoam.tooling.verification.study import Case, Study

_APP = "neofoam solver incompressiblefluid"


def _case() -> Case:
    return Case(
        id="c",
        name="simpleFoam/pitzDaily",
        path=Path("/tut/simpleFoam/pitzDaily"),
        native_solver="simpleFoam",
        app="",
        fields=("U",),
    )


def _study() -> Study:
    return Study(
        title="demo",
        config_path=Path("config.yaml"),
        cases=[_case()],
        tier_titles={},
        config={},
        apps=(_APP,),
    )


def _rundir(cases_root: Path, solver: str) -> Path:
    """The on-disk run dir for the demo case (id ``c``) under *cases_root*."""
    return cases_root / "c" / solver


def test_build_records_stage_failure_instead_of_raising(
    tmp_path: Path, monkeypatch: Any
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("cannot parse controlDict")

    monkeypatch.setattr(runner, "stage", boom)
    stamp = tmp_path / "built.json"

    _build(_study(), _case(), "simpleFoam", tmp_path / "cases", stamp)

    record = json.loads(stamp.read_text())
    assert record["stage_failed"] is True
    assert "cannot parse controlDict" in record["reason"]


def test_swap_neutralises_and_passes_a_failed_build_through(tmp_path: Path) -> None:
    built = tmp_path / "built.json"
    built.write_text(
        json.dumps({"solver": "incompressiblefluid", "stage_failed": True})
    )
    stamp = tmp_path / "swapped.json"

    _swap(_study(), _case(), "incompressiblefluid", tmp_path / "cases", built, stamp)

    assert json.loads(stamp.read_text())["stage_failed"] is True
    # Neutralised: the shell run rule can invoke ./Allrun safely, and it no-ops.
    assert (
        "exit 0"
        in (_rundir(tmp_path / "cases", "incompressiblefluid") / "Allrun").read_text()
    )


def test_swap_records_and_neutralises_no_swap(tmp_path: Path, monkeypatch: Any) -> None:
    def refuse(native: str, app: str) -> Any:
        def step(case: Any) -> None:
            raise NoSwapPoint("no unique solver token in Allrun")

        return step

    monkeypatch.setattr(runner, "swap_solver", refuse)
    built = tmp_path / "built.json"
    built.write_text(json.dumps({"solver": "incompressiblefluid", "case_dir": "d"}))
    stamp = tmp_path / "swapped.json"

    _swap(_study(), _case(), "incompressiblefluid", tmp_path / "cases", built, stamp)

    record = json.loads(stamp.read_text())
    assert record["no_swap"] is True
    assert "no unique solver token" in record["reason"]
    # An un-swappable candidate must NOT run the native solver — Allrun is a no-op.
    assert (
        "exit 0"
        in (_rundir(tmp_path / "cases", "incompressiblefluid") / "Allrun").read_text()
    )


def test_swap_is_a_no_op_for_the_native_solver(tmp_path: Path) -> None:
    built = tmp_path / "built.json"
    built.write_text(json.dumps({"solver": "simpleFoam", "case_dir": "d"}))
    stamp = tmp_path / "swapped.json"

    _swap(_study(), _case(), "simpleFoam", tmp_path / "cases", built, stamp)

    record = json.loads(stamp.read_text())
    assert "no_swap" not in record and "stage_failed" not in record


def _write_swapped(rundir: Path, payload: dict[str, object]) -> None:
    rundir.mkdir(parents=True, exist_ok=True)
    (rundir / ".swapped.json").write_text(json.dumps(payload))


def test_status_from_rundir_reports_stage_failure(tmp_path: Path) -> None:
    rundir = _rundir(tmp_path / "cases", "incompressiblefluid")
    _write_swapped(rundir, {"stage_failed": True, "reason": "boom"})

    status = _status_from_rundir(
        _study(), _case(), "incompressiblefluid", tmp_path / "cases"
    )

    assert status["stage_failed"] is True
    assert status["finished"] is False


def test_status_from_rundir_reports_no_swap(tmp_path: Path) -> None:
    rundir = _rundir(tmp_path / "cases", "incompressiblefluid")
    _write_swapped(rundir, {"no_swap": True, "reason": "no token"})

    status = _status_from_rundir(
        _study(), _case(), "incompressiblefluid", tmp_path / "cases"
    )

    assert status["no_swap"] is True
    assert status["finished"] is False


def test_status_from_rundir_finished_when_log_reached_end(tmp_path: Path) -> None:
    rundir = _rundir(tmp_path / "cases", "incompressiblefluid")
    _write_swapped(rundir, {"case_dir": str(rundir)})
    (rundir / f"log.{_APP}").write_text("Time = 20\nExecutionTime = 1 s\nEnd\n")
    (rundir / ".seconds").write_text("12")

    status = _status_from_rundir(
        _study(), _case(), "incompressiblefluid", tmp_path / "cases"
    )

    assert status["finished"] is True
    assert status["seconds"] == 12.0


def test_status_from_rundir_prefers_a_combined_status_json(tmp_path: Path) -> None:
    """The legacy combined run (packaged driver / VoF) writes a sibling status.json;
    it is authoritative, so one compare function serves both pipelines."""
    cases_root = tmp_path / "cases"
    rundir = _rundir(cases_root, "incompressiblefluid")
    rundir.mkdir(parents=True)
    combined = {"solver": "incompressiblefluid", "finished": True, "seconds": 3.0}
    (rundir.parent / "incompressiblefluid.status.json").write_text(json.dumps(combined))

    status = _status_from_rundir(_study(), _case(), "incompressiblefluid", cases_root)

    assert status == combined  # returned verbatim, no rundir reconstruction


def test_status_from_rundir_extracts_reason_when_no_end(tmp_path: Path) -> None:
    rundir = _rundir(tmp_path / "cases", "incompressiblefluid")
    _write_swapped(rundir, {"case_dir": str(rundir)})
    (rundir / f"log.{_APP}").write_text(
        "Starting time loop\n--> FOAM FATAL ERROR: \nmatrix is singular\n"
    )

    status = _status_from_rundir(
        _study(), _case(), "incompressiblefluid", tmp_path / "cases"
    )

    assert status["finished"] is False
    assert "FOAM FATAL ERROR" in str(status["reason"])


def test_status_from_rundir_falls_back_to_first_token_log(tmp_path: Path) -> None:
    """`application="neofoam solver ..."; runApplication ${application}` leaves
    `${application}` unquoted, so it re-splits and the log is named after the FIRST
    token (`log.neofoam`) — not the full command. The runner must read that fallback
    so the real error surfaces instead of "no solver log written"."""
    rundir = _rundir(tmp_path / "cases", "incompressiblefluid")
    _write_swapped(rundir, {"case_dir": str(rundir)})
    # Only `log.neofoam` exists, not `log.neofoam solver incompressiblefluid`.
    (rundir / "log.neofoam").write_text(
        "Starting time loop\n--> FOAM FATAL ERROR: \nnCorrectors -1\n"
    )

    status = _status_from_rundir(
        _study(), _case(), "incompressiblefluid", tmp_path / "cases"
    )

    assert status["finished"] is False
    assert "FOAM FATAL ERROR" in str(status["reason"])
    assert "no solver log written" not in str(status["reason"])
