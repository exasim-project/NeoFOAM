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

Staging is a :mod:`neofoam.tooling.casebuild` pipeline, so dictionary edits go
through :class:`neofoam.io.DictFile` and address entries *structurally*. That
matters more than it sounds: a regex anchored on ``writeControl`` also rewrites
the ``writeControl`` inside ``functions { ... }``, which turns a sampling
functionObject's ``writeTime`` into ``adjustable`` — it then demands a
``writeInterval`` it does not have and aborts the run in
``system/controlDict/functions``. That single bug produced 26 false native
failures in an earlier sweep.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from neofoam.io import DictFile
from neofoam.tooling.casebuild import CaseDir, Pipeline, Step, from_template, patch

__all__ = [
    "STEP_BUDGET",
    "clean",
    "ensure_allrun",
    "stage",
    "swap_solver",
    "truncate",
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


def truncate(budget: int = STEP_BUDGET) -> Step:
    """Cut the case to *budget* steps and force full-precision output.

    ``writeFormat binary`` is not cosmetic: ASCII at the usual
    ``writePrecision 6`` floors any observable difference at ~1e-6, three orders
    coarser than the round-off this suite is trying to resolve.
    """

    def step(case: CaseDir) -> None:
        control = DictFile(case.path / "system" / "controlDict")
        start = control.get[float]("startTime")
        span = budget * control.get[float]("deltaT")
        patch(
            "system/controlDict",
            stopAt="endTime",
            endTime=start + span,
            writeControl="adjustable",
            writeInterval=span,
            purgeWrite=0,
            writeFormat="binary",
        )(case)

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
