# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Outcome vocabulary, and the readers that say what happened to a run.

Running the tutorial's own ``Allrun`` end to end — rather than pre-splitting it
into mesh and solve stages — is what lets the suite cover an arbitrary tutorial
shape. The ``run`` rule invokes it as plain shell; this module supplies the
outcome codes, the log readers that classify a run afterwards
(:func:`failure_reason`, :func:`log_tail`, :func:`time_dirs`), and
:func:`mesh_fingerprint` — meshing happens twice per case, so a nondeterministic
mesher must be reported as such instead of blamed on the solver.

A solver that crashes is the *finding*, so nothing here raises on a failed run:
the outcome is data, written to disk for the comparison stage to read.
"""

from __future__ import annotations

import hashlib
import re
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
    "MESH_NOT_REPRODUCIBLE",
    "POSTPROCESS_NOT_IMPLEMENTED",
    "failure_reason",
    "log_tail",
    "mesh_fingerprint",
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
#: The case cannot be staged as a drop-in — its ``Allrun`` has no unique solver
#: token to swap the neofoam app in, it drives the solver in a mode neofoam does not
#: implement (a ``-postProcess`` pre-run that seeds ``0/``), or it needs a solver
#: capability neofoam refuses outright (a refining ``dynamicFvMesh``). The case is
#: outside what this study covers, not a solver verdict.
UNSUPPORTED_CASE = "UNSUPPORTED_CASE"
#: Staging the case never got as far as a run — e.g. casebuild could not parse the
#: controlDict (a ``#includeFunc`` it cannot resolve). A harness limitation, not a
#: solver verdict.
CASE_SETUP_FAILED = "CASE_SETUP_FAILED"
COMPARE_FAILED = "COMPARE_FAILED"
#: The two runs finished on identical meshes but ended with different cell counts
#: (internal field lengths disagree — e.g. AMR refined differently), so the final
#: fields are not directly comparable. A real difference between the solvers.
MESH_DIFFERS = "MESH_DIFFERS"
#: The two meshes already disagreed *before* the solve (fingerprint mismatch): the
#: mesher is nondeterministic, so the solvers were never comparable — a harness
#: fault, not a solver verdict.
MESH_NOT_REPRODUCIBLE = "MESH_NOT_REPRODUCIBLE"


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
    meshes = sorted(case.glob("processor*/constant/polyMesh")) or [case / "constant" / "polyMesh"]
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
