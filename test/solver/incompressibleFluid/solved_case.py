# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The solved lid-driven cavity the post-processing end-to-end tests share.

Test-only helper (not shipped). Three e2e modules — this package's
``test_post_process.py`` and ``test/postprocess/test_sources_e2e.py`` /
``test_residuals_e2e.py`` — all need the same case: ``cases/cavity3x3`` under
the :func:`case_presets.lid_driven_cavity` preset, a checked-in declaration
directory copied over it, blockMesh, and the solver run to the end. One copy is
what keeps the three from drifting apart, as they already had over
``writePrecision``. ``test/postprocess`` reaches it as
``solver.incompressibleFluid.solved_case`` — pytest puts ``test/`` on
``sys.path`` for both packages.

**Subprocess per run.** One ``Foam::Time`` per process, so the solver runs in a
fresh interpreter with ``FOAM_SIGFPE`` off, as everywhere else in these tests.

**``writePrecision 12``.** Every e2e expectation is recomputed from the *written*
time directory, so its ASCII round-trip is the whole error budget; twelve digits
leave ~1e-12 relative for the tests to put a tolerance around.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from neofoam.tooling.casebuild import CaseDir, Step, block_mesh, configs, from_template, patch

from .case_presets import lid_driven_cavity

#: ``cases/cavity3x3``: 3x3x1 uniform cells in a 0.1 x 0.1 x 0.01 m box, lid on
#: ``movingWall``. blockMesh numbers its cells with ``x`` fastest.
CAVITY_TEMPLATE = Path(__file__).parent / "cases" / "cavity3x3"

_SOLVER_DRIVER = """
import os
os.environ["FOAM_SIGFPE"] = ""
from neofoam.solver.incompressibleFluid import run
run(["incompressibleFluid"])
"""


def _overlay(source: Path) -> Step:
    """Copy a checked-in declaration directory over the built case."""

    def step(case: CaseDir) -> None:
        shutil.copytree(source, case.path, dirs_exist_ok=True)

    return step


def built_cavity(
    dest: Path,
    *,
    declarations_dir: Optional[Path] = None,
    end_time: float = 0.003,
    delta_t: float = 0.001,
    write_precision: int = 12,
) -> CaseDir:
    """Build :data:`CAVITY_TEMPLATE` at *dest* and overlay *declarations_dir*, unsolved.

    The meshed case :func:`solved_cavity` then runs; separate so a decomposed
    run can put ``decomposePar`` between the two.
    """
    pipeline = (
        from_template(CAVITY_TEMPLATE)
        | configs(*lid_driven_cavity(end_time=end_time, delta_t=delta_t))
        | patch("system/controlDict", writePrecision=write_precision)
    )
    if declarations_dir is not None:
        pipeline = pipeline | _overlay(declarations_dir)
    return (pipeline | block_mesh()).build_at(dest)


def solved_cavity(
    dest: Path,
    *,
    declarations_dir: Optional[Path] = None,
    end_time: float = 0.003,
    delta_t: float = 0.001,
    write_precision: int = 12,
) -> CaseDir:
    """Build :data:`CAVITY_TEMPLATE` at *dest*, overlay *declarations_dir*, solve it.

    Pass the directory holding the case's ``system/postProcess.yaml`` (or
    ``.py``) as *declarations_dir*, or leave it out for a case that declares no
    table. The default three steps of 0.001 s are what the tables are read at.
    """
    case = built_cavity(
        dest,
        declarations_dir=declarations_dir,
        end_time=end_time,
        delta_t=delta_t,
        write_precision=write_precision,
    )
    solve_serially(case)
    return case


def solve_serially(case: CaseDir) -> None:
    """Run ``incompressibleFluid`` to the end in ``case``, in a fresh interpreter."""
    solve = subprocess.run(
        [sys.executable, "-c", _SOLVER_DRIVER],
        cwd=case.path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert solve.returncode == 0, (
        f"incompressibleFluid aborted on {case.path.name}:\n"
        f"{solve.stdout[-2000:]}\n{solve.stderr[-2000:]}"
    )


def read_table(case: CaseDir, name: str) -> tuple[list[str], list[list[str]]]:
    """``postProcessing/<name>.csv`` as its header row and its data rows."""
    with (case.path / "postProcessing" / f"{name}.csv").open(newline="") as handle:
        header, *rows = csv.reader(handle)
    return header, rows
