# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run a staged case's ``Allrun`` and say what happened if it did not finish.

Running the tutorial's own ``Allrun`` end to end — rather than pre-splitting it
into mesh and solve stages — is what lets the suite cover an arbitrary tutorial
shape. The cost is that meshing happens twice per case; because that assumes
meshing is deterministic, :func:`mesh_fingerprint` exists so a nondeterministic
mesher is reported as such instead of being blamed on the solver.

A solver that crashes is the *finding*, so nothing here raises on a failed run:
the outcome is data, written to disk for the comparison stage to read.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

__all__ = [
    "MATCHED",
    "MATCHED_TO_ROUNDOFF",
    "FIELDS_DIFFER",
    "SOLVER_FAILED",
    "NATIVE_FAILED",
    "UNSUPPORTED_CASE",
    "CASE_SETUP_FAILED",
    "COMPARE_FAILED",
    "MESH_DIFFERS",
    "TIMEOUT",
    "POSTPROCESS_NOT_IMPLEMENTED",
    "failure_reason",
    "log_tail",
    "mesh_fingerprint",
    "run_allrun",
    "time_dirs",
]


def log_tail(text: str, lines: int = 200) -> str:
    """The last *lines* lines of a log — enough to carry a crash's traceback."""
    return "\n".join(text.splitlines()[-lines:])


# Outcome codes. NATIVE_FAILED always means *the harness* failed, never the
# solver under test — a study that reports it as a solver defect is lying.
MATCHED = "MATCHED"
#: Every differing field is within ``rel < 1e-10`` — reproduces native to machine
#: precision but not the strict ``rtol=1e-10``/``atol=1e-15``. A separate outcome so a
#: case at round-off (e.g. ``p_rgh`` ~1e4, whose ULP sits above ``atol``) is not read as
#: a failure, while :data:`MATCHED`'s strict semantics stay untouched.
MATCHED_TO_ROUNDOFF = "MATCHED_TO_ROUNDOFF"
#: Ran to completion, but a field differs beyond tolerance. Read ``max rel`` before
#: believing it: at ``atol=1e-15`` a ~1e-13 round-off reads the same as a ~1e0 defect.
FIELDS_DIFFER = "FIELDS_DIFFER"
SOLVER_FAILED = "SOLVER_FAILED"
NATIVE_FAILED = "NATIVE_FAILED"
#: The tutorial's ``Allrun`` has no unique solver token to swap the neofoam app in,
#: so the drop-in variant cannot be staged — the case is outside what this study
#: covers, not a solver verdict.
UNSUPPORTED_CASE = "UNSUPPORTED_CASE"
#: Staging the case never got as far as a run — e.g. casebuild could not parse the
#: controlDict (a ``#includeFunc`` it cannot resolve). A harness limitation, not a
#: solver verdict.
CASE_SETUP_FAILED = "CASE_SETUP_FAILED"
COMPARE_FAILED = "COMPARE_FAILED"
MESH_DIFFERS = "MESH_DIFFERS"
TIMEOUT = "TIMEOUT"


def time_dirs(case: Path) -> list[Path]:
    """Numeric time directories, oldest first."""
    found = []
    for item in case.iterdir():
        if not item.is_dir() or item.name in ("constant", "system"):
            continue
        try:
            float(item.name)
        except ValueError:
            continue
        found.append(item)
    return sorted(found, key=lambda path: float(path.name))


def mesh_fingerprint(case: Path) -> str:
    """Hash the mesh topology, so a nondeterministic mesher cannot look like a bug."""
    digest = hashlib.sha256()
    meshes = sorted(case.glob("processor*/constant/polyMesh")) or [
        case / "constant" / "polyMesh"
    ]
    for mesh in meshes:
        for name in ("owner", "neighbour", "points"):
            path = mesh / name
            if path.is_file():
                digest.update(path.read_bytes())
    return digest.hexdigest()


#: A parallel run that dies this way never reached a timestep — the process tore
#: down MPI and then touched it again. Matched first, because the mpirun teardown
#: banner it produces contains the word "error" and used to shadow the real cause.
_MPI_AFTER_FINALIZE = "MPI_Bcast() function was called after MPI_FINALIZE"

#: A Python exception surfaced through the CLI, e.g. ``NotImplementedError: ...``.
_PY_EXCEPTION = re.compile(r"^\s*(\w*(?:Error|Exception)): (.+)$", re.MULTILINE)

#: The message ``neofoam``'s CLI prints (see ``neofoam.cli.app``) when a solver
#: command is invoked with ``-postProcess``: no neofoam solver can run OpenFOAM's
#: post-processing mode (execute the case's registered function objects without
#: solving) — pybFoam exposes no functionObject-execution binding to build it on.
#: Kept in sync with the CLI's own string by hand (not imported — ``cli.app`` and
#: this module are deliberately not coupled); matched here first so a run that hit
#: it is read as "unsupported mode", not a generic solver crash, and
#: ``runner.py::_decide`` maps it to ``UNSUPPORTED_CASE``.
POSTPROCESS_NOT_IMPLEMENTED = "solver -postProcess mode not implemented"


def failure_reason(log_text: str) -> str:
    """Pull the most explanatory line out of a failed solver log.

    Ordered most-specific first, and every branch is there because a laxer one
    was wrong. A naive scan for ``"error"`` matches mpirun's teardown banner,
    which appears in *every* aborted parallel run and says nothing about why the
    solver stopped; worse, it also matches source lines that ``rich`` renders as
    traceback *context*, which once made 13 cases look like a shared defect that
    did not exist.
    """
    if POSTPROCESS_NOT_IMPLEMENTED in log_text:
        return POSTPROCESS_NOT_IMPLEMENTED

    if _MPI_AFTER_FINALIZE in log_text:
        return "aborted before first timestep: MPI_Bcast after MPI_FINALIZE"

    for marker in ("--> FOAM FATAL IO ERROR", "--> FOAM FATAL ERROR"):
        index = log_text.find(marker)
        if index != -1:
            return " ".join(log_text[index : index + 300].split())

    # rich renders tracebacks inside box-drawing rules; the exception line is the
    # only part worth quoting, and the last one is the actual cause.
    matches = _PY_EXCEPTION.findall(log_text)
    if matches:
        kind, message = matches[-1]
        message = message.rstrip().rstrip("│|").rstrip()  # strip rich's right rule
        return " ".join(f"{kind}: {message}".split())[:300]

    return " ".join(log_text[-300:].split()) or "no solver log written"


def _reached_end(log: Path) -> bool:
    return log.is_file() and "End\n" in log.read_bytes().decode("utf-8", "replace")


def run_allrun(case: Path, solver_log: str, timeout: int = 1800) -> dict[str, object]:
    """Run ``./Allrun`` in *case*; return a status record, never raise.

    *solver_log* is the basename of the log the solver stage writes
    (``log.simpleFoam``, or ``log.<the neofoam command>``). Its trailing ``End``
    is the only reliable signal that the solver actually completed: ``Allrun``
    itself exits 0 even when a stage inside it failed.
    """
    started = time.monotonic()
    try:
        completed = subprocess.run(
            ["./Allrun"],
            cwd=case,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "FOAM_SIGFPE": ""},
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "finished": False,
            "timed_out": True,
            "reason": f"exceeded {timeout}s",
            "seconds": time.monotonic() - started,
        }

    seconds = time.monotonic() - started
    log = case / solver_log
    if _reached_end(log):
        return {"finished": True, "timed_out": False, "reason": "", "seconds": seconds}

    if log.is_file():
        reason = failure_reason(log.read_bytes().decode("utf-8", "replace"))
    else:
        # No solver log at all: the failure is upstream of the solver, in a
        # meshing or setup stage, so Allrun's own output is what explains it.
        reason = " ".join(completed.stdout[-600:].split()) or "no solver log written"
    return {
        "finished": False,
        "timed_out": False,
        "reason": reason,
        "seconds": seconds,
    }
