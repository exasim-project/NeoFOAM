# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``system/fvSolution`` keyed by OpenFOAM regular expressions, on the NeoN path.

OpenFOAM lets the ``solvers`` / ``relaxationFactors`` / ``residualControl`` entries
be keyed by a regex — ``"(U|k|epsilon)"``, ``".*Final"`` — and resolves a field
name against those patterns at lookup time. The NeoN backend copies the dictionary
into NeoN's hash map, so the patterns have to be resolved by NeoFOAM; stored
verbatim (as they were) every per-field lookup missed and the run died with a bare
``IndexError: unordered_map::at`` naming neither the field nor the dictionary.

The case (``cases/regexSolverKeys``) keys everything but the ``p`` solver by regex,
and gives the two solvers different Ginkgo mappings, so the per-solve residual
report — which prints the preconditioner+solver ``fvSolution`` mapped to — shows
*which* entry each field resolved to, not merely that some entry was found.

Runs go through a subprocess: NeoN/Kokkos and OpenFOAM keep per-process global
state that does not survive a second in-process run, and the resolution happens
inside the C++ runtime, so a run is where the selection is observable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from neofoam.tooling.casebuild import block_mesh, from_template, patch

CASES = Path(__file__).parent / "cases"

# Two time steps (deltaT 0.005) — enough to reach the first and the final outer
# corrector, i.e. both the base and the *Final solver lookups.
END_TIME = 0.01


def _run_solver(case_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the framework NeoN solver in ``case_path`` (FOAM_SIGFPE off, as in the cavity run)."""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.incompressibleFluidNeoN import run;"
            " run(['incompressibleFluidNeoN'])",
        ],
        cwd=str(case_path),
        env={**os.environ, "FOAM_SIGFPE": "false"},
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_regex_solver_key_selects_its_settings(tmp_path: Path) -> None:
    """U resolves to the regex entry's solver, p to its own literal entry."""
    case = (
        from_template(CASES / "regexSolverKeys")
        | patch("system/controlDict", endTime=END_TIME)
        | block_mesh()
    ).build_at(tmp_path / "regexSolverKeys")

    result = _run_solver(case.path)

    assert result.returncode == 0, (
        f"regexSolverKeys run failed (rc={result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    )
    # "(U|k|epsilon)" is PBiCGStab + DILU; the literal p entry is PCG + diagonal.
    assert "Ilu+Bicgstab:  Solving for Ux" in result.stdout
    assert "Jacobi+Cg:  Solving for p" in result.stdout


def test_field_without_solver_entry_reports_field_and_available_keys(tmp_path: Path) -> None:
    """Dropping the entry that covers U fails naming U, the dictionary and the keys."""
    case = (
        from_template(CASES / "regexSolverKeys")
        | patch("system/controlDict", endTime=END_TIME)
        | patch("system/fvSolution", remove=["solvers.(U|k|epsilon)"])
        | block_mesh()
    ).build_at(tmp_path / "missingSolverKey")

    result = _run_solver(case.path)

    assert result.returncode != 0, f"run unexpectedly succeeded:\n{result.stdout[-2000:]}"
    output = result.stdout + result.stderr
    assert "FvSolutionKeyNotFound" in output
    assert "No entry for field 'U' in system/fvSolution/solvers" in output
    assert '".*Final", p' in output  # the keys that are left, sorted
    assert "unordered_map::at" not in output
