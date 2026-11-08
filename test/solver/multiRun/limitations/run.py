# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run a limitations case as a normal multiRun case and print what each region ended with.

multiRun finds region solvers by package name (``neofoam.solver.<name>``), so this
runner hands it the test-only ``probe`` solver; otherwise it is ``multiRun.run()``
in the case directory::

    python test/solver/multiRun/limitations/run.py test/solver/multiRun/limitations/cases/pair
"""

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from neofoam.framework.context import Context

MULTI_RUN = importlib.import_module("neofoam.solver.multiRun.multiRun")


def use_probe() -> None:
    """Let multiRun resolve the region solver name ``probe`` to the test-only solver."""
    from solver.multiRun.limitations.probe_solver import probe  # noqa: PLC0415

    resolve = MULTI_RUN.resolve_solver

    def resolve_with_probe(name: str) -> Any:
        return probe if name == "probe" else resolve(name)

    MULTI_RUN.resolve_solver = resolve_with_probe


def run_case(case: Path) -> dict[str, Context]:
    """``multiRun.run()`` in ``case`` with the ``probe`` solver, then back to the old directory."""
    use_probe()
    cwd = Path.cwd()
    os.chdir(case)
    try:
        regions: dict[str, Context] = MULTI_RUN.run()
    finally:
        os.chdir(cwd)
    return regions


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # test/, for solver.multiRun
    for region, region_ctx in run_case(Path(sys.argv[1])).items():
        field = region_ctx.models["probe"].field
        history = region_ctx.models["history"]
        print(f"{region}: {field} = {history[-1]!r} after {len(history)} solves")
