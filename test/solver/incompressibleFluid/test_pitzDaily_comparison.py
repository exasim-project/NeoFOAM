# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field-by-field comparison: incompressibleFluid (PIMPLE) vs native pimpleFoam.

Both solvers run on the bundled ``tutorials/pitzDaily`` case for a short
time interval. The volScalar/volVectorField outputs are loaded from disk
and compared numerically; the solvers should match to round-off.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from neofoam.solver.incompressibleFluid import run

from .comparison_helpers import (
    compare_solver_fields,
    setup_case,
)

FIELDS_TO_COMPARE = [
    ("U", "volVectorField"),
    ("p", "volScalarField"),
    ("k", "volScalarField"),
    ("epsilon", "volScalarField"),
]


def test_pitzDaily_solver_comparison() -> None:
    """Compare incompressibleFluid against native pimpleFoam on pitzDaily."""
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily"

    test_case_custom = repo_root / "test_cases" / "pitzDaily_custom_solver"
    test_case_native = repo_root / "test_cases" / "pitzDaily_native_solver"

    end_time = 0.01
    write_interval = 0.01

    try:
        setup_case(source_case, test_case_custom, end_time, write_interval)
        setup_case(source_case, test_case_native, end_time, write_interval)

        original_dir = Path.cwd()
        os.chdir(test_case_custom)
        try:
            run(["incompressibleFluid"])
        finally:
            os.chdir(original_dir)

        result = subprocess.run(
            ["pimpleFoam", "-case", str(test_case_native)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"pimpleFoam failed: {result.stderr}"

        all_match, failed_fields, failed_details = compare_solver_fields(
            test_case_custom,
            test_case_native,
            FIELDS_TO_COMPARE,
            rtol=1e-10,
            atol=1e-15,
        )

        if not all_match:
            parts = []
            for fname in failed_fields:
                max_abs, max_rel = failed_details.get(
                    fname, (float("nan"), float("nan"))
                )
                parts.append(f"{fname}(abs={max_abs:.3e}, rel={max_rel:.3e})")
            pytest.fail("Field values differ between solvers: " + ", ".join(parts))

    finally:
        for tc in [test_case_custom, test_case_native]:
            if tc.exists():
                shutil.rmtree(tc)
