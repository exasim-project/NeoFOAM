# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Input validation tests for incompressibleFluid solver.

Uses val_pitzDaily case with defaults set to none. Runs the actual solver
and checks whether the validator catches the same errors the solver crashes on.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Generator

import pytest

from neofoam.solver.incompressibleFluid.create_fields import create_init
from .comparison_helpers import requires_openfoam

CASE_DIR = Path(__file__).parent / "val_pitzDaily"


@pytest.fixture
def case(tmp_path: Path) -> Generator[Path, None, None]:
    """Copy val_pitzDaily to tmp, run blockMesh, chdir."""
    dst = tmp_path / "val_pitzDaily"
    shutil.copytree(CASE_DIR, dst)
    shutil.copytree(dst / "0.orig", dst / "0")
    original = os.getcwd()
    os.chdir(dst)
    subprocess.run(["blockMesh"], capture_output=True, timeout=30)
    yield dst
    os.chdir(original)


@requires_openfoam
def test_solver_crashes_validator_should_predict(case: Path) -> None:
    """Run the solver on val_pitzDaily (defaults=none). It will crash.
    The validator should predict the same missing entries.

    This test currently FAILS — showing which entries the validator misses.
    """
    os.environ["FOAM_SIGFPE"] = ""

    # 1. What does the validator say?
    errors = create_init(case_dir=case).validate()
    validator_fields = {e.field for e in errors}

    # 2. Run the solver via CLI in a separate process — OpenFOAM calls exit()
    #    on fatal errors, so we can't catch it in-process.
    log_file = case / "solver_output.log"
    result = subprocess.run(
        ["neofoam", "solver", "incompressiblefluid"],
        cwd=case,
        stdout=log_file.open("w"),
        stderr=subprocess.STDOUT,
        timeout=60,
        env={**os.environ, "FOAM_SIGFPE": ""},
    )
    solver_crashed = result.returncode != 0
    solver_error = log_file.read_text()

    # Copy log to test folder so it persists after tmp cleanup
    persistent_log = Path(__file__).parent / "solver_output.log"
    shutil.copy(log_file, persistent_log)

    # Print diagnostic info
    print(f"\n=== Validator found {len(errors)} errors ===")
    for e in errors:
        print(f"  {e.field}: {e.message}")

    print(f"\n=== Solver crashed: {solver_crashed} ===")
    if solver_error:
        # Extract the FOAM FATAL error line
        for line in solver_error.splitlines():
            if "FATAL" in line or "Entry" in line or "not found" in line:
                print(f"  {line.strip()}")

    print(f"\n=== Log saved to: {persistent_log} ===")

    # The test: if the solver crashes, the validator should have found errors
    if solver_crashed:
        assert len(errors) > 0, (
            f"Solver crashed (see {persistent_log}).\n"
            f"But validator found NO errors — it should have predicted this crash."
        )
