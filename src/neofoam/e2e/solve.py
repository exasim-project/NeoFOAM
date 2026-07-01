# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Stage 4 -- run the ``incompressibleFluid`` solver on the assembled case.

Mirrors the tutorial usage (``chdir`` into the case, then
``incompressibleFluid.run(["."])``). Like the mesh build, the in-process pybFoam
objects SIGBUS at GC teardown, so :func:`main` runs the solve and then
``os._exit``s; the orchestrator and the gated test invoke it as
``python -m neofoam.e2e.solve <case_dir>`` for process isolation.
"""

from __future__ import annotations

from pathlib import Path


def _time_dirs(case_dir: Path) -> list[str]:
    """Written time directories (numeric names other than ``0``), sorted."""
    times = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if value > 0:
            times.append(child.name)
    return sorted(times, key=float)


def run_solver(case_dir: str | Path, log_file: str | Path | None = None):  # type: ignore[no-untyped-def]
    """Run one ``incompressibleFluid`` simulation in ``case_dir``; return its Context.

    Requires pybFoam + a sourced OpenFOAM (imported lazily). The caller is
    responsible for process teardown (see :func:`main`).
    """
    import contextlib

    from neofoam.solver.incompressibleFluid import run

    with contextlib.chdir(case_dir):
        return run(["."], log_file=log_file)


def main(argv: list[str] | None = None) -> int:
    """CLI: run the solver in an isolated process (``os._exit`` past GC teardown)."""
    import argparse
    import os
    import sys
    import traceback

    parser = argparse.ArgumentParser(description="Run incompressibleFluid on a case.")
    parser.add_argument("case_dir", help="Assembled case directory.")
    parser.add_argument("--log", default=None, help="Redirect solver stdout to this file.")
    args = parser.parse_args(argv)
    case = Path(args.case_dir)
    try:
        run_solver(case, log_file=args.log)
        written = _time_dirs(case)
        if not written:
            print("solver produced no time directories", file=sys.stderr)
            sys.stderr.flush()
            os._exit(1)
        print(f"solver wrote time dirs: {written}", file=sys.stderr)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
