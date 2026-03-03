# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Comparison test for hotRoom case: NeoFOAM solver vs native buoyantBoussinesqPimpleFoam.

Runs a single timestep and compares the log output of both solvers to verify
that the same equations are being solved with matching residuals.
"""

import os
import re
import subprocess
from pathlib import Path

from neofoam.solver.incompressibleFluid import run

from .comparison_helpers import (
    requires_openfoam,
    setup_case,
)

# Disable OpenFOAM floating point exception trapping BEFORE any imports
os.environ["FOAM_SIGFPE"] = ""


def _extract_solving_lines(log: str) -> list[str]:
    """Extract 'Solving for ...' lines from solver log output."""
    return [line.strip() for line in log.splitlines() if "Solving for" in line]


def _extract_residual_fields(log: str) -> list[str]:
    """Extract field names from 'Solving for <field>' lines."""
    pattern = re.compile(r"Solving for (\w+),")
    return pattern.findall(log)


@requires_openfoam
def test_hotRoom_solver_comparison() -> None:
    """Compare NeoFOAM solver against native buoyantBoussinesqPimpleFoam on hotRoom (1 timestep)."""

    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "hotRoom"

    test_case_custom = repo_root / "test_cases" / "hotRoom_custom_solver"
    test_case_native = repo_root / "test_cases" / "hotRoom_native_solver"

    # Single timestep: deltaT=2, endTime=2
    end_time = 1000.0
    write_interval = 200.0

    try:
        # Setup both cases
        setup_case(
            source_case, test_case_custom, end_time, write_interval, run_setfields=True
        )
        setup_case(
            source_case, test_case_native, end_time, write_interval, run_setfields=True
        )

        # Run custom solver, pipe output to log file
        original_dir = Path.cwd()
        os.chdir(test_case_custom)
        custom_log = test_case_custom / "solver.log"
        try:
            run(["simpleSolver"], log_file=custom_log)
        finally:
            os.chdir(original_dir)

        custom_output = custom_log.read_text() if custom_log.exists() else ""

        # Run native buoyantBoussinesqPimpleFoam, pipe output to log file
        native_log = test_case_native / "solver.log"
        with open(native_log, "w") as f:
            result = subprocess.run(
                ["buoyantBoussinesqPimpleFoam", "-case", str(test_case_native)],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
        assert result.returncode == 0, (
            f"buoyantBoussinesqPimpleFoam failed with returncode {result.returncode}"
        )
        native_output = native_log.read_text()

        # Extract fields solved by each solver
        custom_fields = _extract_residual_fields(custom_output)
        native_fields = _extract_residual_fields(native_output)

        # The custom solver must solve equations (non-empty)
        assert len(custom_fields) > 0, (
            f"Custom solver did not solve any equations.\nLog output:\n{custom_output}"
        )

        # Both solvers should solve the same fields in the same order
        assert custom_fields == native_fields, (
            f"Solved fields differ.\n"
            f"Custom: {custom_fields}\n"
            f"Native: {native_fields}\n"
            f"\nCustom log:\n{custom_output}\n"
            f"\nNative log:\n{native_output}"
        )

        # Both solvers should produce the same number of solving lines
        custom_solving = _extract_solving_lines(custom_output)
        native_solving = _extract_solving_lines(native_output)

        assert len(custom_solving) == len(native_solving), (
            f"Number of solving lines differs.\n"
            f"Custom ({len(custom_solving)} lines):\n"
            + "\n".join(custom_solving)
            + f"\n\nNative ({len(native_solving)} lines):\n"
            + "\n".join(native_solving)
        )

    finally:
        import shutil

        for test_case in [test_case_custom, test_case_native]:
            if test_case.exists():
                shutil.rmtree(test_case)
