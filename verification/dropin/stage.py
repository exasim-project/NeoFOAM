# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Build one side of a comparison from a pristine tutorial.

Both sides are staged from the tutorial itself, never by cloning the finished
native run. Cloning is unsound: tutorials mix guarded and unguarded commands, and
``runApplication`` skips any stage whose ``log.<app>`` exists. In
``simpleFoam/motorBike`` the unguarded ``restore0Dir -processor`` re-runs and
resets the fields while the guarded ``potentialFoam`` is skipped, so the second
solver silently starts from different initial conditions — which once showed up
as a ``FIELDS_DIFFER`` on a case independently proven bit-identical.

A tutorial's dictionaries are never rewritten in place. Two ways of editing them
are ruled out, each by a bug it caused. A regex anchored on ``writeControl`` also
rewrites the ``writeControl`` inside ``functions { ... }``, which turns a sampling
functionObject's ``writeTime`` into ``adjustable`` — it then demands a
``writeInterval`` it does not have and aborts the run in
``system/controlDict/functions``. That single bug produced 26 false native
failures in an earlier sweep. And reading the file through
:class:`neofoam.io.DictFile` to edit it *structurally* expands every ``#include``
/ ``#sinclude`` / ``${...}`` directive at read time and writes back the expansion,
which either aborts the interpreter with an unresolvable-include FOAM fatal or
silently bakes in an empty ``#sinclude``. So the controlDict is only ever
*appended to* — see :func:`truncate`.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from neofoam.tooling.casebuild import CaseDir, Pipeline, Step, from_template
from verification.dropin.foamdict import entry, read

__all__ = [
    "MESH_REFINEMENT",
    "POSTPROCESS_PRE_STEP",
    "STEP_BUDGET",
    "clean",
    "ensure_allrun",
    "needs_postprocess",
    "stage",
    "swap_solver",
    "truncate",
    "uses_mesh_refinement",
]

#: Every case is truncated to this many steps of its own ``deltaT``.
#:
#: For a case with ``adjustTimeStep yes`` this bounds *simulated time* at
#: ``STEP_BUDGET * deltaT``, which the solver may cross in fewer, larger steps.
#: Both sides get the identical controlDict, so the comparison stays fair — but
#: a ``MATCHED`` proves algorithm parity over a short window, not long-run
#: stability, and a report must say so.
STEP_BUDGET = 20


def clean() -> Step:
    """Drop run artifacts a tutorial may ship, so ``Allrun`` starts from zero.

    ``runApplication``/``runParallel`` skip any stage whose ``log.<app>`` already
    exists. A tutorial that ships committed logs (``interIsoFoam/damBreak`` does)
    therefore has *every* stage skipped and produces no result at all — which
    reads as a solver failure when it is nothing of the kind.
    """

    def step(case: CaseDir) -> None:
        for log in case.path.glob("log.*"):
            log.unlink()
        for directory in [*case.path.glob("processor*"), case.path / "postProcessing"]:
            shutil.rmtree(directory, ignore_errors=True)
        for item in case.path.iterdir():
            if not item.is_dir() or item.name in ("constant", "system"):
                continue
            try:
                written = float(item.name)
            except ValueError:
                continue
            if written > 0.0:  # written times only, never the initial conditions
                shutil.rmtree(item, ignore_errors=True)

    return step


#: The truncation, re-stated below the tutorial's own entries. OpenFOAM merges a
#: repeated keyword by replacing the earlier one, so appending is enough to
#: override — and it leaves every byte above it, directives included, untouched.
_TRUNCATION = """

// --- appended by the drop-in verification harness ---------------------------
// Overrides, not edits: everything above is the tutorial's own text (parsing and
// re-writing a controlDict resolves away its preprocessor directives), and
// OpenFOAM's last-definition-wins merge makes these entries the effective ones.
stopAt          endTime;
endTime         {end!r};
writeControl    adjustable;
writeInterval   {span!r};
purgeWrite      0;
writeFormat     binary;
"""


def _scalar(text: str, key: str) -> float:
    """A numeric controlDict entry, read without parsing the dictionary."""
    value = entry(text, key)
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"controlDict has no numeric {key} (read {value!r})") from None


def truncate(budget: int = STEP_BUDGET) -> Step:
    """Cut the case to *budget* steps and force full-precision output.

    ``writeFormat binary`` is not cosmetic: ASCII at the usual
    ``writePrecision 6`` floors any observable difference at ~1e-6, three orders
    coarser than the round-off this suite is trying to resolve.

    Written as an appended override block rather than an edit, and read with the
    static :mod:`~verification.dropin.foamdict` reader rather than a real
    dictionary parse, because a tutorial's controlDict may hold directives that
    only resolve inside a running case (``#includeFunc graphs``,
    ``#sinclude "<constant>/dynamicMeshDict"``, ``${FOAM_EXECUTABLE}``). Parsing
    it here resolves them against the wrong context — fatally, or silently to
    nothing — and re-serialising then makes that loss permanent.
    """

    def step(case: CaseDir) -> None:
        control = case.path / "system" / "controlDict"
        text = read(control)
        start = _scalar(text, "startTime")
        span = budget * _scalar(text, "deltaT")
        # Appended as bytes: some tutorials ship dictionaries that are not valid
        # UTF-8, and a decode/encode round-trip would rewrite those bytes.
        with control.open("ab") as out:
            out.write(_TRUNCATION.format(end=start + span, span=span).encode())

    return step


#: What ``bin/foamRunTutorials`` runs when a tutorial ships no ``Allrun``.
_FALLBACK_ALLRUN = """#!/bin/sh
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions
#------------------------------------------------------------------------------
[ -d 0 ] || restore0Dir

runApplication blockMesh

runApplication $(getApplication)
"""


def ensure_allrun() -> Step:
    """Write the standard fallback ``Allrun`` when the tutorial ships none.

    Not an invention: ``bin/foamRunTutorials`` applies exactly this recipe
    (``blockMesh`` plus the controlDict's ``application``) to any case without an
    ``Allrun``, so generating it keeps the sweep faithful to how OpenFOAM itself
    would run the tutorial.
    """

    def step(case: CaseDir) -> None:
        allrun = case.path / "Allrun"
        if allrun.is_file():
            return
        allrun.write_text(_FALLBACK_ALLRUN)
        allrun.chmod(0o755)

    return step


#: Why a case whose ``Allrun`` runs the solver in post-processing mode cannot be
#: compared as a drop-in. ``interIsoFoam/notchedDiscInSolidBodyRotation`` runs
#: ``${application} -postProcess -time 0`` before the solve; that pass executes the
#: case's ``setFlow`` function object, which *writes* the prescribed velocity into
#: ``0/U`` and ``0/phi``. No neofoam solver implements that mode (see
#: ``neofoam.cli.app``), and the tutorial ``Allrun``s have no ``set -e``, so the
#: pre-step's non-zero exit is swallowed and the solve starts from an unseeded
#: ``0/``. The resulting field difference is this gap, not a discretisation
#: difference — so the case is reported as out of scope rather than compared.
POSTPROCESS_PRE_STEP = (
    "Allrun seeds 0/ via a `-postProcess` pre-run that neofoam does not implement"
)


#: A shell variable assignment (``application=$(getApplication)``), which names the
#: solver without running it — so it is not the script's first solver invocation.
_ASSIGNMENT = re.compile(r"\s*[A-Za-z_]\w*=")


def needs_postprocess(case: Path, native: str) -> bool:
    """Whether the staged ``Allrun``'s *first* solver run is a ``-postProcess`` one.

    The rule is positional, and deliberately narrow: only a ``-postProcess``
    invocation that runs **before** the real solve seeds fields the solve then
    needs, and only that makes the case un-comparable as a drop-in.
    ``interIsoFoam/notchedDiscInSolidBodyRotation`` is the motivating case — its
    ``-postProcess`` pass writes ``0/U`` and ``0/phi`` before the solver runs.

    A tutorial may just as well run the solver in ``-postProcess`` mode *after*
    the solve, as analysis over the finished result
    (``pimpleFoam/laminar/cylinder2D`` samples ``(U p)`` that way). That pass seeds
    nothing and cannot change the compared fields, so the case stays in scope: a
    bare substring test refused it, and lost a bit-for-bit ``MATCHED``.

    Checked on the *staged* script rather than declared per case, so any tutorial
    that adopts the idiom is caught by the same rule. Read before the solver token
    is swapped, so the solver is still spelled the tutorial's own way: literally,
    as ``$(getApplication)``, or through the ``${application}`` variable that
    carries either. Backslash continuations are joined first — a tutorial wraps a
    single invocation over two lines.
    """
    allrun = case / "Allrun"
    if not allrun.is_file():
        return False
    invocation = re.compile(
        rf"\$\(getApplication\)|\$\{{application\}}|\$application(?![\w])"
        rf"|(?<![\w./-]){re.escape(native)}(?![\w-])"
    )
    for line in allrun.read_text().replace("\\\n", " ").splitlines():
        if _ASSIGNMENT.match(line) or not invocation.search(line):
            continue
        return "-postProcess" in line  # the first solver run decides
    return False


#: Why a case whose mesh refines itself cannot be compared as a drop-in. The
#: native solver refines and unrefines around the interface, so its mesh at the
#: write time has a different cell count than the one it started from; neofoam
#: implements mesh *motion* but not topology change and refuses such a case (see
#: ``neofoam.foam.initialization.create_mesh``). Comparing anyway would only
#: report the two runs' cell counts disagreeing — the gap, not a numeric verdict.
MESH_REFINEMENT = (
    "constant/dynamicMeshDict selects a refining dynamicFvMesh (AMR), which neofoam "
    "does not implement"
)


def uses_mesh_refinement(case: Path) -> bool:
    """Whether the staged case selects a topology-changing (AMR) ``dynamicFvMesh``.

    Read from the *staged* dictionary rather than declared per case, so any tutorial
    that selects a refinement mesh is caught by the same rule. Mesh motion is not
    caught: only a type that refines cells (``dynamicRefineFvMesh`` and friends).
    """
    return "Refine" in entry(read(case / "constant" / "dynamicMeshDict"), "dynamicFvMesh")


class NoSwapPoint(Exception):
    """``Allrun`` has no single, unambiguous solver token to replace."""


def swap_solver(native: str, app: str) -> Step:
    """Replace the solver invocation in ``Allrun`` with the neofoam command.

    Tutorials spell the solver either literally or as ``$(getApplication)``.
    ``$(getApplication)`` cannot be redirected by editing controlDict's
    ``application`` entry, because the substitution is unquoted: a multi-word
    value re-splits and ``runParallel`` emits ``neofoam -parallel solver ...``.
    The token itself has to go, quoted so it survives as one word.

    Three idioms are handled:

    * ``$(getApplication)`` (or the literal solver name) appearing **once** — swap it.
    * The same token appearing **more than once, identically** (a commented-out serial
      variant, or a ``-s restart`` second run) — swap *every* occurrence, so both runs
      use the same candidate. This is unambiguous because all occurrences are the same
      token; picking one would only be a problem if they differed.
    * ``application="<solver>"`` combined with ``${application}`` — the token lives in a
      shell-variable assignment, so the *assignment's value* is swapped (a bare-token
      substitution would nest quotes: ``application=""neofoam ...""``).

    A tutorial with no recognizable solver invocation is refused, not guessed at.
    """

    getapp = re.compile(r"\$\(getApplication\)")
    literal = re.compile(rf"(?<![\w./-]){re.escape(native)}(?![\w-])")
    assignment = re.compile(rf'(\bapplication=)"?{re.escape(native)}"?(?![\w./-])')

    def step(case: CaseDir) -> None:
        allrun = case.path / "Allrun"
        text = allrun.read_text()

        # `application="<solver>"; ... ${application}`: swap the assignment's value.
        uses_var = "${application}" in text or "$application" in text
        if uses_var and len(assignment.findall(text)) == 1:
            allrun.write_text(assignment.sub(rf'\1"{app}"', text))
            return

        # `$(getApplication)` or the literal solver, once or repeated-identically:
        # swap every occurrence (they are the same token, so this is unambiguous).
        for pattern in (getapp, literal):
            if pattern.findall(text):
                allrun.write_text(pattern.sub(f'"{app}"', text))
                return

        msg = f"no unique solver token in Allrun (looked for $(getApplication), {native})"
        raise NoSwapPoint(msg)

    return step


def stage(
    tutorial: Path,
    dest: Path,
    *,
    native: str,
    app: str = "",
    budget: int = STEP_BUDGET,
    extra: tuple[Step, ...] = (),
) -> CaseDir:
    """Materialize one side of the comparison at *dest*.

    With *app* empty this is the native side and ``Allrun`` is left untouched;
    with *app* set the solver token is swapped. *extra* steps are applied to this
    side only, after truncation — the one sanctioned way to deviate from a pure
    drop-in (e.g. injecting an ``advectionScheme`` key), and any study that uses
    it must say so in its report.
    """
    pipeline: Pipeline = from_template(tutorial) | clean() | truncate(budget)
    for step in extra:
        pipeline = pipeline | step
    pipeline = pipeline | ensure_allrun()
    if app:
        pipeline = pipeline | swap_solver(native, app)
    return pipeline.build_at(dest)
