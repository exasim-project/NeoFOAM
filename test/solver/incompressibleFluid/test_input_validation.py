# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Input validation contract test for incompressibleFluid.

Runs the solver on the ``val_pitzDaily`` case (intentionally incomplete
fvSolution / turbulenceProperties). If the solver crashes, the validator
should have predicted the same missing entries upfront. Ported from
``feat/python_solvers``; the only API adaptation is that validation is
now reached via ``runner.run_load().validate()`` rather than a direct
``.validate()`` on the staged-init runner.

The source-branch test is documented as a "currently FAILS" contract
test exposing gaps in the validator — porting it preserves that intent.
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
    """Copy val_pitzDaily to tmp, restore 0/, run blockMesh, chdir."""
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
    """The solver crashes on val_pitzDaily; the validator should predict it.

    Currently EXPECTED TO FAIL — exposes validator gaps in the minimal
    port (no FvSchemes/FvSolution/transportProperties configs are
    loaded into LoadResult yet, so ``validate()`` finds zero errors).
    """
    os.environ["FOAM_SIGFPE"] = ""

    load_result = create_init(case_dir=case).run_load()
    errors = load_result.validate()

    log_file = case / "solver_output.log"
    with log_file.open("w") as f:
        result = subprocess.run(
            ["neofoam", "solver", "incompressiblefluid"],
            cwd=case,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=60,
            env={**os.environ, "FOAM_SIGFPE": ""},
        )
    solver_crashed = result.returncode != 0

    persistent_log = Path(__file__).parent / "solver_output.log"
    shutil.copy(log_file, persistent_log)

    print(f"\n=== Validator found {len(errors)} errors ===")
    for e in errors:
        print(f"  {e.field}: {e.message}")

    print(f"\n=== Solver crashed: {solver_crashed} ===")

    if solver_crashed:
        assert len(errors) > 0, (
            f"Solver crashed (see {persistent_log}).\n"
            f"But validator found NO errors — it should have predicted this crash."
        )
