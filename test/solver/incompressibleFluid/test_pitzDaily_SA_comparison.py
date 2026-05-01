# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Comparison test for pitzDaily_SA case: NeoFOAM solver vs native pimpleFoam.
Tests that both solvers produce matching results for Spalart-Allmaras turbulence.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from neofoam.solver.incompressibleFluid import run

from .comparison_helpers import (
    requires_openfoam,
    setup_case,
    compare_solver_fields,
)

# Disable OpenFOAM floating point exception trapping
os.environ["FOAM_SIGFPE"] = ""

# SA uses nuTilda instead of k/epsilon
FIELDS_TO_COMPARE = [
    ("U", "volVectorField"),
    ("p", "volScalarField"),
    ("nuTilda", "volScalarField"),
    ("nut", "volScalarField"),
]


@requires_openfoam
def test_pitzDaily_SA_solver_comparison() -> None:
    """Compare NeoFOAM solver against native pimpleFoam on pitzDaily SA case."""

    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily_SA"

    test_case_custom = repo_root / "test_cases" / "pitzDaily_SA_custom_solver"
    test_case_native = repo_root / "test_cases" / "pitzDaily_SA_native_solver"

    end_time = 0.01
    write_interval = 0.01

    try:
        # Setup both cases
        print("\n=== Setting up pitzDaily_SA test cases ===")
        setup_case(source_case, test_case_custom, end_time, write_interval)
        setup_case(source_case, test_case_native, end_time, write_interval)

        # Run NeoFOAM solver
        print("\n=== Running NeoFOAM solver (SA) ===")
        original_dir = Path.cwd()
        os.chdir(test_case_custom)
        try:
            run(["simpleSolver"])
        finally:
            os.chdir(original_dir)

        # Run native pimpleFoam
        print("\n=== Running native pimpleFoam (SA) ===")
        result = subprocess.run(
            ["pimpleFoam", "-case", str(test_case_native)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"pimpleFoam failed: {result.stderr}"

        # Compare fields
        all_match, failed_fields, failed_details = compare_solver_fields(
            test_case_custom,
            test_case_native,
            FIELDS_TO_COMPARE,
            rtol=1e-10,
            atol=1e-15,
        )

        if not all_match:
            detail_parts = []
            for field_name in failed_fields:
                max_abs_diff, max_rel_diff = failed_details.get(
                    field_name, (float("nan"), float("nan"))
                )
                detail_parts.append(
                    f"{field_name}(max_abs={max_abs_diff:.3e}, "
                    f"max_rel={max_rel_diff:.3e})"
                )
            failure_msg = (
                "SA field values differ between solvers. Failed fields: "
                + ", ".join(detail_parts)
            )
            assert False, failure_msg

        print("\n=== Test PASSED: SA results match exactly ===")

    finally:
        for test_case in [test_case_custom, test_case_native]:
            if test_case.exists():
                shutil.rmtree(test_case)
                print(f"Cleaned up: {test_case}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
