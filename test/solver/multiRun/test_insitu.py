# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The case's in-situ entries draw every region of a multiRun together.

``cases/two_regions`` (two uncoupled slabs, 10 steps of 0.1) with an overlay:
``insitu`` declares a ``regions_recorder`` entry every second step at the case
root; ``insitu_failing`` one that raises; ``insitu_in_region`` an entry in a
region's own ``system/postProcess.yaml``; ``insitu_table`` a table at the case
root. ``regions_recorder`` (:mod:`.insitu_recorder`) records what it was handed.
Under ``mpirun`` each region runs on a process of its own and hands in its own
region only.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _mpirun import mpirun, needs_mpi

from neofoam.framework.context import Context
from neofoam.solver.multiRun import run
from solver.multiRun.case_files import copy_case
from solver.multiRun.insitu_recorder import CALLS

HERE = Path(__file__).parent
WORKER = HERE / "insitu_worker.py"
RESULT_PREFIX = "#RESULT "


@pytest.fixture(autouse=True)
def _no_calls() -> None:
    CALLS.clear()


def _run(tmp_path: Path, overlay: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Context]:
    monkeypatch.chdir(copy_case("two_regions", tmp_path / "case", overlay))
    return run()


def test_both_regions_are_drawn_together_on_the_entry_s_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(tmp_path, "insitu", monkeypatch)

    drawn = [regions for regions, _ in CALLS[:-1]]
    assert [sorted(regions) for regions in drawn] == [["left", "right"]] * 5
    assert [regions["left"][0] for regions in drawn] == [regions["right"][0] for regions in drawn]
    assert CALLS[-1] == "finalize"


def test_a_frame_shows_the_step_just_solved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = _run(tmp_path, "insitu", monkeypatch)

    (last, _) = CALLS[-2]
    assert last["left"] == (10, pytest.approx(float(final["left"].fields["T"].mean())))
    assert last["right"] == (10, pytest.approx(float(final["right"].fields["T"].mean())))


def test_one_region_thread_draws_each_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(tmp_path, "insitu", monkeypatch)

    assert {thread for _, thread in CALLS[:-1]} <= {"left", "right"}


def test_a_failing_entry_fails_the_run_and_no_region_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="the renderer broke"):
        _run(tmp_path, "insitu_failing", monkeypatch)


def test_a_region_s_own_insitu_entry_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="region left declares insitu entries"):
        _run(tmp_path, "insitu_in_region", monkeypatch)


def test_a_table_at_the_case_root_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="insitu entries only"):
        _run(tmp_path, "insitu_table", monkeypatch)


@needs_mpi
def test_under_mpi_each_process_hands_in_its_own_region_at_the_same_steps(
    tmp_path: Path,
) -> None:
    case = copy_case("two_regions", tmp_path / "case", "insitu")
    for region in ("left", "right"):  # one process each; the slab reads no mesh
        (case / region / "processor0").mkdir()
    path = [str(HERE.parent.parent), os.environ.get("PYTHONPATH", "")]
    done = subprocess.run(
        [str(mpirun()), "-np", "2", sys.executable, str(WORKER)],
        cwd=case,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(p for p in path if p),
            "OMPI_MCA_rmaps_base_oversubscribe": "1",
        },
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    calls = sorted(
        (json.loads(line[len(RESULT_PREFIX) :]) for line in done.stdout.splitlines()
         if line.startswith(RESULT_PREFIX)),
        key=lambda c: c["region"],
    )  # fmt: skip

    assert [c["region"] for c in calls] == ["left", "right"]
    for c in calls:
        other = "right" if c["region"] == "left" else "left"
        drawn = [regions for regions, _ in c["calls"][:-1]]
        assert len(drawn) == 5 and all(regions[other] is None for regions in drawn)
        assert [regions[c["region"]][0] for regions in drawn] == [2, 4, 6, 8, 10]
        assert c["calls"][-1] == "finalize"
