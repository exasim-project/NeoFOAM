# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""decomposePar / mpirun / reconstructPar plumbing for the VoF parallel tests.

The steps every decomposed run needs, kept here so the test modules stay a
setup -> act -> assert read. ``mpirun``, ``decomposePar`` and ``reconstructPar``
are hard dependencies exactly like ``interFoam`` (TEST_STYLE: no
skip-if-missing) — an environment without MPI fails these tests loudly rather
than reporting a green suite that proved nothing about parallel operation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent.parent

# Two subdomains, split along y — see the file's own comment for why that split
# and not an x-split. Cases that ship their own system/decomposeParDict (the
# pressure-reference row) do not use it.
DECOMPOSE_PAR_DICT = _HERE / "_parallel_decomposeParDict"
PARALLEL_DRIVER = _HERE / "_parallel_driver.py"
NPROCS = 2


def install_decompose_par_dict(case: Path) -> None:
    """Give ``case`` the shared two-subdomain decomposeParDict."""
    shutil.copyfile(DECOMPOSE_PAR_DICT, case / "system" / "decomposeParDict")


def decompose(case: Path) -> None:
    """Split ``case`` into per-rank sub-cases with its own decomposeParDict."""
    result = subprocess.run(
        ["decomposePar", "-case", str(case)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"decomposePar failed for {case}:\n{result.stdout[-4000:]}\n{result.stderr}"
    )


def run_mpi(command: list[str], case: Path, nprocs: int = NPROCS) -> str:
    """Run ``command`` under ``mpirun -np <nprocs>`` in ``case``; return stdout.

    ``cwd=case`` rather than ``-case <case>``: the solver re-reads
    ``system/controlDict`` relative to the working directory every step (the
    ``runTimeModifiable`` handling in ``set_time_step``), so a parallel run has
    to be launched from the case root, just as ``Allrun`` does.
    """
    result = subprocess.run(
        ["mpirun", "-np", str(nprocs), *command],
        cwd=case,
        capture_output=True,
        text=True,
        # Bounded low on purpose: the runs are seconds, and a rank-local
        # reduction shows up as a *deadlock* (one rank inside a collective the
        # other never enters), which otherwise hangs the suite.
        timeout=300,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
    )
    assert result.returncode == 0, (
        f"parallel run {command} failed in {case}:\n"
        f"stdout:\n{result.stdout[-6000:]}\nstderr:\n{result.stderr[-4000:]}"
    )
    return result.stdout


def run_parallel_solver(case: Path, nprocs: int = NPROCS) -> str:
    """Run incompressibleVoF across ``nprocs`` ranks in ``case``; return stdout."""
    return run_mpi([sys.executable, str(PARALLEL_DRIVER)], case, nprocs)


def rank_case(case: Path, rank: int, dest: Path) -> Path:
    """Copy one rank's sub-case out as a standalone case at ``dest``.

    ``processor<rank>/`` carries its own ``constant/polyMesh`` and time
    directories but no ``system/`` — decomposePar leaves the dictionaries in the
    parent — so the two halves are recombined here. Lets the per-rank fields be
    read with the same OpenFOAM reader every other comparison uses, instead of
    parsing the written files by hand.
    """
    shutil.copytree(case / f"processor{rank}", dest)
    shutil.copytree(case / "system", dest / "system")
    return dest


def reconstruct(case: Path) -> None:
    """Merge the per-rank time directories back into the case root."""
    result = subprocess.run(
        ["reconstructPar", "-case", str(case)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"reconstructPar failed for {case}:\n{result.stdout[-4000:]}\n{result.stderr}"
    )
