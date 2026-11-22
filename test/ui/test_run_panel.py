# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Running a case from the Review step: the Allrun process and the logs it writes."""

from __future__ import annotations

import asyncio
import os
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("trame")

from trame.app import get_server  # noqa: E402

from neofoam.ui.run_panel import TAIL_LINES, RunPanel, newest_log, read_appended  # noqa: E402


@pytest.fixture
def panel(request: pytest.FixtureRequest) -> Any:
    """A RunPanel on a bare server named after the requesting test."""
    return RunPanel(get_server(f"neofoam_run_{request.node.name}"))


def _allrun(case: Path, body: str) -> Path:
    case.mkdir(parents=True, exist_ok=True)
    script = case / "Allrun"
    script.write_text('#!/bin/sh\ncd "${0%/*}" || exit 1\n' + textwrap.dedent(body))
    script.chmod(0o755)
    return script


# -- the two log helpers ------------------------------------------------------


def test_newest_log_picks_the_most_recently_written(tmp_path: Path) -> None:
    for name, when in (("log.blockMesh", 1000.0), ("log.snappyHexMesh", 2000.0)):
        (tmp_path / name).write_text("x")
        os.utime(tmp_path / name, (when, when))

    assert newest_log(tmp_path, since=0.0) == tmp_path / "log.snappyHexMesh"


def test_newest_log_ignores_a_previous_runs_leftover(tmp_path: Path) -> None:
    # wab-transient already holds log.blockMesh from an earlier run; showing it as
    # this run's output would be a lie.
    stale = tmp_path / "log.blockMesh"
    stale.write_text("from yesterday")
    os.utime(stale, (1000.0, 1000.0))

    assert newest_log(tmp_path, since=5000.0) is None


def test_newest_log_of_a_case_without_logs_is_none(tmp_path: Path) -> None:
    (tmp_path / "Allrun").write_text("")
    assert newest_log(tmp_path, since=0.0) is None


def test_read_appended_returns_only_what_is_new(tmp_path: Path) -> None:
    log = tmp_path / "log.solver"
    log.write_text("one\ntwo\n")

    text, offset = read_appended(log, 0)
    assert text == "one\ntwo\n"

    log.write_text("one\ntwo\nthree\n")
    more, offset = read_appended(log, offset)
    assert more == "three\n"


def test_read_appended_restarts_when_the_file_was_truncated(tmp_path: Path) -> None:
    # runApplication rewrites a log; resuming at the old offset would read nothing.
    log = tmp_path / "log.solver"
    log.write_text("a long previous run\n")
    _, offset = read_appended(log, 0)

    log.write_text("new\n")
    text, _ = read_appended(log, offset)
    assert text == "new\n"


# -- running ------------------------------------------------------------------


def test_run_streams_the_scripts_own_stdout(panel, tmp_path: Path) -> None:
    # A wizard-scaffolded Allrun execs the solver: stdout is all there is.
    _allrun(tmp_path, "echo hello-from-allrun\n")
    panel._server.state.target_dir = str(tmp_path)  # noqa: SLF001

    asyncio.run(panel.run_case())

    assert "hello-from-allrun" in panel._server.state.run_lines  # noqa: SLF001
    assert panel._server.state.run_severity == "success"  # noqa: SLF001
    assert "exit 0" in panel._server.state.run_status  # noqa: SLF001


def test_run_tails_the_log_files_the_script_writes(panel, tmp_path: Path) -> None:
    # An OpenFOAM-convention Allrun is silent on stdout and writes log.* instead.
    _allrun(
        tmp_path,
        """
        echo meshing > log.blockMesh
        sleep 0.6
        echo solving > log.incompressibleFluid
        sleep 0.6
        """,
    )
    panel._server.state.target_dir = str(tmp_path)  # noqa: SLF001

    asyncio.run(panel.run_case())

    lines = panel._server.state.run_lines  # noqa: SLF001
    assert "meshing" in lines
    assert "solving" in lines
    # Each log is announced as it becomes the one being followed.
    assert "— log.blockMesh —" in lines
    assert "— log.incompressibleFluid —" in lines


def test_no_output_is_lost_when_the_run_moves_to_the_next_log(panel, tmp_path: Path) -> None:
    # Found by running it: a step writes its closing lines after the poll that saw it
    # last, and the poller had already moved on to the next log by the time it read.
    _allrun(
        tmp_path,
        """
        echo one   >> log.blockMesh
        sleep 0.5
        echo two   >> log.blockMesh
        echo three >> log.blockMesh
        echo start >> log.incompressibleFluid
        sleep 0.5
        """,
    )
    panel._server.state.target_dir = str(tmp_path)  # noqa: SLF001

    asyncio.run(panel.run_case())

    lines = panel._server.state.run_lines  # noqa: SLF001
    assert [ln for ln in lines if ln in {"one", "two", "three"}] == ["one", "two", "three"]
    assert "start" in lines


def test_a_failing_allrun_is_reported_as_an_error(panel, tmp_path: Path) -> None:
    _allrun(tmp_path, "echo nope >&2\nexit 3\n")
    panel._server.state.target_dir = str(tmp_path)  # noqa: SLF001

    asyncio.run(panel.run_case())

    assert panel._server.state.run_severity == "error"  # noqa: SLF001
    assert "exited with 3" in panel._server.state.run_status  # noqa: SLF001
    assert "nope" in panel._server.state.run_lines  # noqa: SLF001


def test_clean_runs_the_allclean_script(panel, tmp_path: Path) -> None:
    _allrun(tmp_path, "echo unused\n")
    clean = tmp_path / "Allclean"
    clean.write_text('#!/bin/sh\ncd "${0%/*}" || exit 1\nrm -f log.*\necho cleaned\n')
    clean.chmod(0o755)
    (tmp_path / "log.blockMesh").write_text("from the last run\n")
    panel._server.state.target_dir = str(tmp_path)  # noqa: SLF001

    asyncio.run(panel.clean_case())

    assert "cleaned" in panel._server.state.run_lines  # noqa: SLF001
    assert panel._server.state.run_severity == "success"  # noqa: SLF001
    assert not list(tmp_path.glob("log.*"))


def test_clean_without_an_allclean_names_that_script(panel, tmp_path: Path) -> None:
    _allrun(tmp_path, "echo only-allrun-here\n")  # Allrun present, Allclean is not
    panel._server.state.target_dir = str(tmp_path)  # noqa: SLF001

    asyncio.run(panel.clean_case())

    assert "No Allclean in" in panel._server.state.run_status  # noqa: SLF001


def test_run_without_a_saved_case_says_so(panel, tmp_path: Path) -> None:
    panel._server.state.target_dir = str(tmp_path)  # noqa: SLF001 - no Allrun in it

    asyncio.run(panel.run_case())

    assert panel._server.state.run_severity == "error"  # noqa: SLF001
    assert "save the case first" in panel._server.state.run_status  # noqa: SLF001


def test_run_needs_an_absolute_target(panel) -> None:
    panel._server.state.target_dir = "relative/case"  # noqa: SLF001

    asyncio.run(panel.run_case())

    assert panel._server.state.run_severity == "error"  # noqa: SLF001
    assert "absolute path" in panel._server.state.run_status  # noqa: SLF001


def test_the_window_keeps_only_the_tail_of_a_long_run(panel, tmp_path: Path) -> None:
    # A snappyHexMesh log runs to tens of thousands of lines and the whole state is
    # pushed to the browser on every poll.
    _allrun(tmp_path, f"seq 1 {TAIL_LINES * 3}\n")
    panel._server.state.target_dir = str(tmp_path)  # noqa: SLF001

    asyncio.run(panel.run_case())

    lines = panel._server.state.run_lines  # noqa: SLF001
    assert len(lines) <= TAIL_LINES
    assert lines[-1] == str(TAIL_LINES * 3)  # the newest survive, not the oldest


def test_a_second_run_starts_from_an_empty_window(panel, tmp_path: Path) -> None:
    _allrun(tmp_path, "echo first\n")
    panel._server.state.target_dir = str(tmp_path)  # noqa: SLF001
    asyncio.run(panel.run_case())

    _allrun(tmp_path, "echo second\n")
    asyncio.run(panel.run_case())

    lines = panel._server.state.run_lines  # noqa: SLF001
    assert "second" in lines
    assert "first" not in lines
