# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

""" "Run case": execute the case's ``Allrun`` and tail what it produces.

Two kinds of case reach this panel and they report progress differently, so both are
followed at once:

* a wizard-scaffolded ``Allrun`` (:mod:`neofoam.ui.scaffold`) ``exec``s the solver, so
  everything it has to say arrives on **stdout**;
* an OpenFOAM-convention ``Allrun`` uses ``runApplication``, which is near-silent on
  stdout and writes ``log.blockMesh`` / ``log.snappyHexMesh`` / ``log.<solver>``
  **files** in the case directory instead.

So the process' own output is streamed *and* the case directory is watched for
``log.*`` files, the most recently written one being the one shown — which follows a
multi-step ``Allrun`` from meshing through to the solve on its own.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from neofoam.ui._paths import _resolve_target

__all__ = ["RunPanel", "newest_log", "read_appended"]

#: Lines kept in the window. A snappyHexMesh log runs to tens of thousands; the state
#: is pushed to the browser on every poll, so the whole of one cannot go in it.
TAIL_LINES = 400

#: How often the logs are re-read and the window repainted.
_POLL_S = 0.4

#: A log that predates the run is a previous run's leftover, not this one's output.
_STALE_S = 2.0


def newest_log(case: Path, since: float) -> Path | None:
    """The ``log.*`` file in ``case`` written most recently, ignoring older ones.

    ``since`` is when the run started: an untouched ``log.blockMesh`` from yesterday
    would otherwise be picked up and shown as if this run had produced it.
    """
    try:
        logs = [p for p in case.glob("log.*") if p.is_file()]
    except OSError:
        return None
    fresh = [(p.stat().st_mtime, p) for p in logs if p.stat().st_mtime >= since - _STALE_S]
    return max(fresh, default=(0.0, None))[1] if fresh else None


def read_appended(path: Path, offset: int) -> tuple[str, int]:
    """Text of ``path`` past ``offset``, with the offset to resume from.

    A file that shrank (a re-run truncated it) is read from the start again.
    """
    try:
        size = path.stat().st_size
        with path.open("r", errors="replace") as handle:
            handle.seek(0 if size < offset else offset)
            return handle.read(), handle.tell()
    except OSError:
        return "", offset


class RunPanel:
    """The Review step's run controls: start ``Allrun``, stop it, show its output."""

    def __init__(self, server: Any) -> None:
        self._server = server
        self._state = server.state
        self._proc: Any = None
        self._lines: list[str] = []
        state = server.state
        state.run_busy = False
        state.run_lines = []
        state.run_status = ""
        state.run_severity = "info"
        state.run_source = ""
        server.controller.run_case = self.run_case
        server.controller.clean_case = self.clean_case
        server.controller.stop_run = self.stop_run

    # -- controllers -----------------------------------------------------------

    async def run_case(self) -> None:
        """Execute ``<case>/Allrun`` and follow it until it exits."""
        await self._start("Allrun")

    async def clean_case(self) -> None:
        """Execute ``<case>/Allclean`` — drops the time directories, logs and mesh."""
        await self._start("Allclean")

    async def _start(self, script_name: str) -> None:
        """Run ``<case>/<script_name>``, streaming it into the one output window."""
        state = self._state
        # The button is disabled while busy, but a queued click still lands here.
        if state.run_busy:
            return
        try:
            case = _resolve_target(state.target_dir, "target directory")
        except ValueError as exc:
            self._report(str(exc), "error")
            return
        script = case / script_name
        if not script.is_file():
            self._report(f"No {script_name} in {case} — save the case first.", "error")
            return
        self._lines = []
        with state:  # flush now — the window has to clear before the first await
            state.run_lines = []
            state.run_source = ""
            state.run_busy = True
            state.run_status = f"Running {script}"
            state.run_severity = "info"
        try:
            await self._execute(case, script)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the wizard
            self._report(f"Could not run {script_name}: {exc}", "error")
        finally:
            with state:
                state.run_busy = False

    def stop_run(self) -> None:
        """Terminate the running script (its children keep OpenFOAM's own exit)."""
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            self._append("— stopped —")
            self._flush()

    # -- the run ---------------------------------------------------------------

    async def _execute(self, case: Path, script: Path) -> None:
        """Start ``script`` in ``case`` and pump its output until it exits."""
        started = time.time()
        self._proc = await asyncio.create_subprocess_exec(
            str(script),
            cwd=str(case),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # The process' own output and the log files it writes are two independent
        # sources; reading stdout blocks until a line arrives, so the poller that
        # repaints the window cannot be the same task.
        pipe = asyncio.create_task(self._drain(self._proc))
        try:
            await self._poll(case, started)
        finally:
            pipe.cancel()
            code = self._proc.returncode
            self._pump(case, started)  # whatever landed after the last poll
            ok = code == 0
            self._report(
                f"Finished — exit {code}" if ok else f"{script.name} exited with {code}",
                "success" if ok else "error",
            )
            self._proc = None

    async def _drain(self, proc: Any) -> None:
        """Copy the process' stdout into the buffer, line by line, until it closes."""
        assert proc.stdout is not None
        async for raw in proc.stdout:
            self._append(raw.decode(errors="replace").rstrip("\n"))

    async def _poll(self, case: Path, started: float) -> None:
        """Repaint the window every ``_POLL_S`` while the process runs."""
        assert self._proc is not None
        while self._proc.returncode is None:
            await asyncio.sleep(_POLL_S)
            self._pump(case, started)
        await self._proc.wait()

    def _pump(self, case: Path, started: float) -> None:
        """Take whatever the newest log file has added, then repaint."""
        log = newest_log(case, started)
        if log is not None and str(log) != self._state.run_source:
            if self._state.run_source:
                # The step that just finished wrote its closing lines after the last
                # poll; take them before following the next log, or they are lost.
                self._take(Path(self._state.run_source))
            self._state.run_source = str(log)
            self._offset = 0
            self._append(f"— {log.name} —")
        if log is not None:
            self._take(log)
        self._flush()

    def _take(self, path: Path) -> None:
        """Append whatever ``path`` has grown by since the last read."""
        text, self._offset = read_appended(path, self._offset)
        for line in text.splitlines():
            self._append(line)

    # -- state -----------------------------------------------------------------

    _offset: int = 0

    def _append(self, line: str) -> None:
        """Add one line to the buffer, keeping only the tail of a long run."""
        self._lines.append(line)
        if len(self._lines) > TAIL_LINES:
            del self._lines[: len(self._lines) - TAIL_LINES]

    def _flush(self) -> None:
        """Push the buffer to the browser (a new list — trame compares by reference)."""
        with self._state:
            self._state.run_lines = list(self._lines)

    def _report(self, message: str, severity: str) -> None:
        with self._state:
            self._state.run_status = message
            self._state.run_severity = severity

    # -- rendering -------------------------------------------------------------

    def render(self, v3: Any, html: Any) -> None:
        """The Run button, its status and the tailed output."""
        ctrl = self._server.controller
        with v3.VRow(align="center", classes="mb-2", no_gutters=True):
            v3.VBtn(
                "Run case",
                click=ctrl.run_case,
                color="primary",
                variant="flat",
                prepend_icon="mdi-play",
                loading=("run_busy",),
                disabled=("!target_dir.trim() || run_busy",),
            )
            v3.VBtn(
                "Clean case",
                click=ctrl.clean_case,
                variant="tonal",
                classes="ml-3",
                prepend_icon="mdi-broom",
                disabled=("!target_dir.trim() || run_busy",),
            )
            v3.VBtn(
                "Stop",
                click=ctrl.stop_run,
                color="error",
                variant="tonal",
                classes="ml-3",
                prepend_icon="mdi-stop",
                v_show="run_busy",
            )
            v3.VSpacer()
            html.Div(
                "{{ run_source.split('/').pop() }}",
                classes="text-body-2 text-medium-emphasis",
                v_show="run_source",
            )
        v3.VAlert(
            text=("run_status",),
            type=("run_severity",),
            variant="tonal",
            classes="mb-3",
            v_show="run_status",
        )
        # column-reverse keeps the scroll pinned to the newest line without any JS,
        # so the rows are rendered newest-first inside it.
        with html.Div(
            v_show="run_lines.length",
            classes="pa-3 mb-3 rounded",
            style=(
                "max-height: 420px; overflow-y: auto; display: flex;"
                " flex-direction: column-reverse; background: rgba(127,127,127,0.10);"
                " font-family: ui-monospace, monospace; font-size: 12px;"
                " white-space: pre-wrap;"
            ),
        ):
            with html.Div():
                html.Div("{{ line }}", v_for="(line, i) in run_lines", key="i")
