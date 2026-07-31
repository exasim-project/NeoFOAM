# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The per-case workers the packaged ``study_*.smk`` rules shell out to.

Mirrors :mod:`neofoam.tooling.workflow.sweep_runner`: the Snakemake rules stay
one-line ``shell:`` bodies and the real work is a subcommand here, so the rule
graph is parallel and resumable and the logic is testable without Snakemake. The
pipeline's three per-run rules map to::

    python -m verification.dropin.runner build   --solver <label> ...
    python -m verification.dropin.runner swap    --solver <label> ...
    ./Allrun                                     # the run rule is plain shell

``build`` stages a native-ready case (applying the config's per-case ``simplify:``
patch, which every side gets); ``swap`` swaps the candidate's solver token, applies
the candidate-only ``neo_patch``, and neutralises a case that cannot run; the shell
``run`` rule just executes ``./Allrun``. Nothing here interprets the run —
``compare`` reads each run dir (:func:`_status_from_rundir`) to decide
finished/why-not, diffs every candidate against the native reference into one
``results/<id>.json``, and ``report`` folds those into HTML.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from neofoam.tooling.casebuild import CaseDir, Step, patch
from verification.dropin.cases import Case, Study, load_study
from verification.dropin.compare import FieldDiff, compare_runs
from verification.dropin.execute import (
    CASE_SETUP_FAILED,
    NATIVE_FAILED,
    POSTPROCESS_NOT_IMPLEMENTED,
    SOLVER_FAILED,
    UNSUPPORTED_CASE,
    failure_reason,
    log_tail,
)
from verification.dropin.report import harness_faults, render_report
from verification.dropin.stage import (
    MESH_REFINEMENT,
    POSTPROCESS_PRE_STEP,
    NoSwapPoint,
    needs_postprocess,
    stage,
    swap_solver,
    uses_mesh_refinement,
)

__all__ = ["main"]


def _case_dir(work: Path, case: Case, label: str) -> Path:
    return work / case.id / label


def _neo_patch(study: Study) -> tuple[Step, ...]:
    """Extra staging steps that deviate the neo side only, from ``neo_patch``.

    A study may declare ``neo_patch: {rel/dict/path: {key: value, ...}}`` to inject
    keys the neofoam solver needs but no upstream tutorial ships (e.g.
    ``advectionScheme isoAdvector`` for the interIsoFoam fallback). Absent or empty
    ⇒ no extra steps, i.e. byte-for-byte the pure drop-in — so a study without the
    key is unaffected.
    """
    neo_patch = study.config.get("neo_patch") or {}
    return tuple(patch(rel, **overrides) for rel, overrides in neo_patch.items())


def _simplify(study: Study, case: Case) -> tuple[Step, ...]:
    """Staging steps that simplify this case, from the config's ``simplify:`` channel.

    A study may declare a per-case ``simplify:`` entry to substitute settings the
    neofoam solver does not support (e.g. plain SIMPLE for SIMPLEC). Unlike
    ``neo_patch`` these run in :func:`_build`, which every side goes through, so the
    native reference and every candidate solve the *same* simplified case and a match
    still means something. No entry ⇒ no extra steps, i.e. the untouched drop-in.
    """
    simplification = study.simplify(case.name)
    patches: dict[str, dict[str, object]] = simplification.get("patch", {})
    return tuple(patch(rel, overrides) for rel, overrides in patches.items())


def _mark_failed(solver: str, status_out: Path) -> None:
    """Write a CASE_SETUP_FAILED status for a ``run`` that died before it could.

    The rule's ``|| mark-failed`` backstop: the runner can be hard-exited by a
    FOAM fatal error during staging, which no Python ``except`` can catch, so the
    only place left to record it is a separate process. The one-line reason points
    at the snakemake log, which captured the actual FOAM error.
    """
    status = {
        "solver": solver,
        "finished": False,
        "no_swap": False,
        "stage_failed": True,
        "reason": "staging crashed the runner (uncatchable FOAM error — see snakemake log)",
    }
    status_out.parent.mkdir(parents=True, exist_ok=True)
    status_out.write_text(json.dumps(status, indent=2))


# --- The three-rule split: build -> swap -> run -------------------------------
#
# `build` and `swap` are Python (casebuild staging + the solver-token swap); `run`
# is a plain shell rule that just executes the prepared `./Allrun` — no timeout, no
# interpretation. All the "did it finish, and why not" logic lives in `_compare`,
# which reads the run dir (`_status_from_rundir`) instead of a status file the run
# step used to write. A case that cannot run (staging crashed, or no unique solver
# token) is *neutralised* by `swap` — its `Allrun` is replaced with a no-op — so the
# shell `run` is always safe to invoke and never runs an un-swapped native solver in
# a candidate's dir (which would score a false MATCHED). The reason is carried in
# the swap stamp and read back at compare time.


def _candidate_log_names(app: str) -> list[str]:
    """Solver-log basenames a candidate run may have written, most specific first.

    ``runApplication`` names its log after the *first word* of what it runs. A quoted
    swap (``runParallel "neofoam solver incompressiblevof"``) survives as one word, so
    the log is ``log.neofoam solver incompressiblevof``. But the
    ``application="neofoam solver incompressiblevof"; runApplication ${application}``
    idiom leaves ``${application}`` unquoted — it re-splits and the log is named after
    the first token only: ``log.neofoam``. Try both so a real error surfaces either way.
    """
    names = [f"log.{app}"]
    first = f"log.{app.split()[0]}"
    if first not in names:
        names.append(first)
    return names


def _resolve_solver_log(run_dir: Path, case: Case, app: str, native: bool) -> Path:
    """The solver log to read from *run_dir*: an existing one, else the primary name."""
    if native:
        return run_dir / f"log.{case.native_solver}"
    candidates = _candidate_log_names(app)
    for name in candidates:
        path = run_dir / name
        if path.is_file():
            return path
    return run_dir / candidates[0]


def _neutralize(case_dir: Path) -> None:
    """Replace a non-runnable case's ``Allrun`` with a no-op that exits 0.

    So the shell ``run`` rule can invoke ``./Allrun`` unconditionally: a case that
    failed staging or has no solver token to swap produces no result and no native
    run leaks into a candidate dir. The *why* is in ``.swapped.json``.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    allrun = case_dir / "Allrun"
    allrun.write_text("#!/bin/sh\n# not runnable — see .swapped.json for the reason\nexit 0\n")
    allrun.chmod(0o755)


def _build(study: Study, case: Case, solver: str, cases_root: Path, stamp: Path) -> None:
    """Stage a native-ready case dir (identical recipe for every solver).

    No solver swap and no ``neo_patch`` here — those are the candidate deviation,
    applied by :func:`_swap`. The config's ``simplify:`` patch is the opposite kind
    of deviation and so belongs here: every side is built by this function, so both
    the reference and the candidates get the same simplified case. Records a terminal
    ``stage_failed`` stamp instead of raising, so one unparsable case never aborts
    the DAG.
    """
    case_dir = _case_dir(cases_root, case, solver)
    shutil.rmtree(case_dir, ignore_errors=True)  # a rerun must start clean
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"solver": solver, "case_dir": str(case_dir)}
    try:
        stage(case.path, case_dir, native=case.native_solver, app="")
        staged = CaseDir(case_dir)
        for step in _simplify(study, case):
            step(staged)
    except Exception as exc:
        reason = str(exc).strip().splitlines()[-1] if str(exc).strip() else repr(exc)
        record = {"solver": solver, "stage_failed": True}
        record["reason"] = f"staging failed: {reason}"[:300]
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(record, indent=2))


def _out_of_scope(case_dir: Path, native: str) -> str:
    """Why this staged case cannot be compared as a drop-in at all, or ``""``.

    Both reasons are properties of the case, not of a run: the candidate would
    otherwise "finish" and the comparison would score the missing capability as a
    field or mesh difference, which reads as a numeric verdict it is not. Asked
    before the solver token is swapped, because :func:`needs_postprocess` reads the
    ``Allrun`` and needs the tutorial's own spelling of the solver to find its
    first run.
    """
    if needs_postprocess(case_dir, native):
        return POSTPROCESS_PRE_STEP
    if uses_mesh_refinement(case_dir):
        return MESH_REFINEMENT
    return ""


def _swap(
    study: Study, case: Case, solver: str, cases_root: Path, built: Path, stamp: Path
) -> None:
    """Make the built case run this solver: candidate-only swap + ``neo_patch``.

    Native is a no-op (its ``Allrun`` stays pristine). A terminal build stamp is
    passed straight through; ``NoSwapPoint`` becomes a ``no_swap`` stamp. Never
    raises — the outcome is carried to the run stage as data.

    A case :func:`_out_of_scope` names — an ``Allrun`` that seeds ``0/`` with a
    ``-postProcess`` pre-run, or a mesh that refines itself — is refused the same
    way, so it reports as ``UNSUPPORTED_CASE`` instead of being run and scored.
    """
    case_dir = _case_dir(cases_root, case, solver)
    prior = json.loads(built.read_text())
    if prior.get("stage_failed"):
        _neutralize(case_dir)  # nothing to run; the shell `run` no-ops safely
        stamp.write_text(json.dumps(prior, indent=2))  # passthrough
        return

    record: dict[str, object] = {"solver": solver, "case_dir": str(case_dir)}
    if solver != case.native_label:
        cd = CaseDir(case_dir)
        for step in _neo_patch(study):
            step(cd)
        reason = _out_of_scope(case_dir, case.native_solver)
        if not reason:
            try:
                swap_solver(case.native_solver, study.candidates[solver])(cd)
            except NoSwapPoint as exc:
                reason = str(exc)
        if reason:
            record = {"solver": solver, "no_swap": True, "reason": reason}
            _neutralize(case_dir)  # never run the un-swapped native in a candidate dir
    stamp.write_text(json.dumps(record, indent=2))


def _status_from_rundir(
    study: Study, case: Case, solver: str, cases_root: Path
) -> dict[str, object]:
    """Reconstruct one solver's run outcome from its dir — read, don't run.

    The shell ``run`` rule only executes ``./Allrun``; this is where "did it finish,
    and why not" is decided, at compare time. A terminal swap stamp
    (``stage_failed``/``no_swap``) short-circuits; otherwise completion is the
    solver log's trailing ``End`` (``Allrun`` exits 0 even when a stage failed), and
    the reason comes from that log, or the ``Allrun`` output when no solver log was
    written. Returns the same dict shape :func:`_decide` consumes.
    """
    case_dir = _case_dir(cases_root, case, solver)
    swapped = json.loads((case_dir / ".swapped.json").read_text())
    status: dict[str, object] = {"solver": solver, "case_dir": str(case_dir)}

    if swapped.get("stage_failed"):
        status.update(finished=False, no_swap=False, stage_failed=True)
        status.update(reason=swapped.get("reason", ""), seconds=0.0)
        return status
    if swapped.get("no_swap"):
        status.update(finished=False, no_swap=True, seconds=0.0)
        status["reason"] = swapped.get("reason", "")
        return status

    native = solver == case.native_label
    app = "" if native else study.candidates[solver]
    solver_log = _resolve_solver_log(case_dir, case, app, native)
    seconds = _read_seconds(case_dir)
    log_text = solver_log.read_bytes().decode("utf-8", "replace") if solver_log.is_file() else ""
    if "End\n" in log_text:
        status.update(finished=True, no_swap=False, reason="")
    elif log_text:
        status.update(finished=False, no_swap=False)
        status["reason"] = failure_reason(log_text)
    else:
        # No solver log: the failure is upstream in Allrun (meshing/setup), so its
        # captured output is what explains it.
        allrun = case_dir / "log.allrun"
        out = allrun.read_bytes().decode("utf-8", "replace") if allrun.is_file() else ""
        status.update(finished=False, no_swap=False)
        status["reason"] = " ".join(out[-600:].split()) or "no solver log written"
    status["seconds"] = seconds
    return status


def _read_seconds(case_dir: Path) -> float:
    """The run's wall-clock seconds, written by the shell ``run`` rule; 0 if absent."""
    sec_file = case_dir / ".seconds"
    if not sec_file.is_file():
        return 0.0
    try:
        return float(sec_file.read_text().strip())
    except ValueError:
        return 0.0


def _decide(
    native: dict[str, Any],
    neo: dict[str, Any],
    native_dir: Path,
    neo_dir: Path,
    fields: list[str],
) -> tuple[str, str, list[FieldDiff]]:
    """Fold the two side statuses (+ a field diff when both ran) into one outcome."""
    # A staging failure on either side is a harness limitation — surface it before
    # anything else, so it is never mistaken for a native or solver run failure.
    if native.get("stage_failed") or neo.get("stage_failed"):
        failed = native if native.get("stage_failed") else neo
        return CASE_SETUP_FAILED, str(failed.get("reason", "")), []
    if not native.get("finished"):
        # NATIVE_FAILED is always a harness fault, never the solver under test.
        return NATIVE_FAILED, str(native.get("reason", "")), []
    if neo.get("no_swap"):
        return UNSUPPORTED_CASE, str(neo.get("reason", "")), []
    if not neo.get("finished"):
        reason = str(neo.get("reason", ""))
        if reason == POSTPROCESS_NOT_IMPLEMENTED:
            # The candidate's Allrun invoked ``-postProcess``, which no neofoam
            # solver implements — a harness/coverage limitation, not a solver
            # crash, so it reads UNSUPPORTED_CASE rather than SOLVER_FAILED with a
            # raw Allrun trace.
            return UNSUPPORTED_CASE, reason, []
        return SOLVER_FAILED, reason, []
    return compare_runs(native_dir, neo_dir, fields)


# Which run's Allrun log explains a given failure. NATIVE_FAILED is a harness
# fault (the reference never finished); the rest are the candidate solver's.
_FAIL_SIDE = {NATIVE_FAILED: "native", SOLVER_FAILED: "candidate"}


def _failing_log(
    study: Study, case: Case, work: Path, outcome: str, candidate: str
) -> tuple[str, str]:
    """The crashed run's solver-log path (relative) and its tail, or ``("", "")``.

    Read straight from the run directory, which survives for exactly the cases
    that failed — so the report can attach the real log, not just the one-line
    reason. Returns empty strings when the outcome is not a crash.
    """
    side = _FAIL_SIDE.get(outcome)
    if side is None:
        return "", ""
    if side == "native":
        run_dir = _case_dir(work, case, case.native_label)
        log_file = _resolve_solver_log(run_dir, case, "", native=True)
    else:
        run_dir = _case_dir(work, case, candidate)
        log_file = _resolve_solver_log(run_dir, case, study.candidates[candidate], native=False)
    if not log_file.is_file():
        return "", ""
    tail = log_tail(log_file.read_bytes().decode("utf-8", "replace"))
    return log_file.as_posix(), tail


def _compare(study: Study, case: Case, work: Path, out: Path) -> None:
    """Diff every candidate backend against the shared native reference.

    Native runs once per case; each candidate is decided against it, so the
    record carries one entry per backend under ``candidates``.
    """
    native = _status_from_rundir(study, case, case.native_label, work)
    native_dir = _case_dir(work, case, case.native_label)

    candidates = []
    for label in study.candidate_labels:
        status = _status_from_rundir(study, case, label, work)
        cand_dir = _case_dir(work, case, label)
        outcome, detail, diffs = _decide(native, status, native_dir, cand_dir, list(case.fields))
        log_path, log = _failing_log(study, case, work, outcome, label)
        # Report the worst *disagreement*, i.e. over the fields that did not match —
        # the same set `_classify` decides the outcome from. Taken over every field,
        # a matched field's round-off could exceed the real disagreement and the
        # headline number would then contradict the verdict beside it.
        disagreeing = [d for d in diffs if not d.matched]
        candidates.append(
            {
                "label": label,
                "app": study.candidates[label],
                "outcome": outcome,
                "detail": detail,
                "worst_abs": max((d.abs_diff for d in disagreeing), default=0.0),
                "worst_rel": max((d.rel_diff for d in disagreeing), default=0.0),
                "fields": [d.as_dict() for d in diffs],
                "log_path": log_path,
                "log_tail": log,
                "seconds": status.get("seconds", 0.0),
            }
        )

    record = {
        "id": case.id,
        "name": case.name,
        "native_solver": case.native_solver,
        "tier": case.tier,
        "predicted_blocker": case.reason,
        "turbulence": case.turbulence,
        "parallel": case.parallel,
        "native_seconds": native.get("seconds", 0.0),
        # The study identity behind this result, so a results dir that mixes two
        # sweeps (e.g. the strict config vs a neo_patch variant) cannot masquerade as
        # one: each record names the config it came from and the exact neo-side patch.
        "study_config": study.config_path.name,
        "neo_patch": study.config.get("neo_patch") or {},
        # The both-sides substitution this case was run with, empty when pristine —
        # so a MATCHED verdict is read against the case that actually ran.
        "simplify": study.simplify(case.name),
        "candidates": candidates,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))


def _report(study: Study, results_dir: Path, out: Path, diagnostic: bool = False) -> int:
    """Render the report; a harness fault in the record set fails the step.

    A fault (``NATIVE_FAILED``, ``CASE_SETUP_FAILED``, ``COMPARE_FAILED``,
    ``MESH_NOT_REPRODUCIBLE``) means the harness — not the solver under test —
    broke, so the sweep is not a clean three-state result and must not pass
    silently. The report is still written first, so the fault section is there
    to read; the faulted cases are named on stderr.
    """
    records = [json.loads(path.read_text()) for path in sorted(results_dir.glob("*.json"))]
    out.write_text(render_report(study, records, diagnostic=diagnostic))
    faults = harness_faults(study, records)
    for row in faults:
        detail = row.get("detail", "")
        line = f"harness fault: {row['name']} [{row.get('label', '')}] {row['outcome']}"
        print(f"{line}: {detail}" if detail else line, file=sys.stderr)
    return 1 if faults else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verification.runner")
    parser.add_argument("--config", required=True, type=Path, help="study config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    # `build` -> `swap` are the Python half of the three-rule split; the third rule,
    # `run`, is a plain shell `./Allrun` and so has no subcommand here.
    build = sub.add_parser("build", help="stage a native-ready case dir")
    build.add_argument("--case", required=True, help="case id")
    build.add_argument("--solver", required=True, help="solver label")
    build.add_argument("--cases", required=True, type=Path, help="cases root dir")
    build.add_argument("--stamp", required=True, type=Path, help=".built.json to write")

    swap = sub.add_parser("swap", help="swap the solver + apply neo_patch (candidate)")
    swap.add_argument("--case", required=True, help="case id")
    swap.add_argument("--solver", required=True, help="solver label")
    swap.add_argument("--cases", required=True, type=Path, help="cases root dir")
    swap.add_argument("--built", required=True, type=Path, help=".built.json to read")
    swap.add_argument("--stamp", required=True, type=Path, help=".swapped.json to write")

    # `build` stages in-process, and casebuild can hard-exit the whole interpreter
    # on a FOAM fatal error (e.g. an unresolvable controlDict `#includeFunc`) —
    # uncatchable in Python. The rule invokes it as `build ... || mark-failed ...`,
    # so a crashed build still leaves a CASE_SETUP_FAILED stamp and the DAG
    # completes instead of one bad case stranding the whole report.
    marked = sub.add_parser("mark-failed", help="write a CASE_SETUP_FAILED status")
    marked.add_argument("--solver", required=True, help="solver label")
    marked.add_argument("--status", required=True, type=Path, help="status JSON to write")

    compare = sub.add_parser("compare", help="diff a case's two runs")
    compare.add_argument("--case", required=True, help="case id")
    compare.add_argument("--cases", required=True, type=Path, help="cases root dir")
    compare.add_argument("--out", required=True, type=Path, help="results JSON to write")

    report = sub.add_parser("report", help="render report.html from results/")
    report.add_argument("--results", required=True, type=Path, help="results dir")
    report.add_argument("--out", required=True, type=Path, help="report.html to write")
    report.add_argument(
        "--diagnostic",
        action="store_true",
        help="add the per-row max abs/max rel columns and the outcome legend",
    )

    args = parser.parse_args(argv)

    # Dispatched before load_study on purpose: this is the crash fallback, so it
    # must not itself depend on anything that could be the thing that crashed.
    if args.command == "mark-failed":
        _mark_failed(args.solver, args.status)
        return 0

    study = load_study(args.config)

    if args.command == "build":
        _build(study, study.by_id(args.case), args.solver, args.cases, args.stamp)
    elif args.command == "swap":
        _swap(
            study,
            study.by_id(args.case),
            args.solver,
            args.cases,
            args.built,
            args.stamp,
        )
    elif args.command == "compare":
        _compare(study, study.by_id(args.case), args.cases, args.out)
    else:
        return _report(study, args.results, args.out, args.diagnostic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
