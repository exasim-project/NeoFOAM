# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Comparison test for hotRoom case: Buoyant solver vs native buoyantBoussinesqPimpleFoam.
Tests that both solvers produce matching results by loading and comparing
the actual volScalarField/volVectorField data from disk.
"""

import os
import subprocess
from pathlib import Path

from neofoam.solver.incompressibleFluid import run

from .comparison_helpers import (
    requires_openfoam,
    setup_case,
    compare_solver_fields,
)

# Disable OpenFOAM floating point exception trapping BEFORE any imports
os.environ["FOAM_SIGFPE"] = ""

# Fields to compare between solvers: (field_name, field_type)
FIELDS_TO_COMPARE = [
    ("U", "volVectorField"),
    ("p", "volScalarField"),
    ("T", "volScalarField"),
    ("p_rgh", "volScalarField"),
    ("k", "volScalarField"),
    ("epsilon", "volScalarField"),
    ("alphat", "volScalarField"),
]


@requires_openfoam
def test_hotRoom_solver_comparison() -> None:
    """Compare SimpleSolver against native buoyantBoussinesqPimpleFoam on hotRoom case."""

    # Setup paths
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "hotRoom"

    test_case_custom = repo_root / "test_cases" / "hotRoom_custom_solver"
    test_case_native = repo_root / "test_cases" / "hotRoom_native_solver"

    end_time = 1000.0
    write_interval = 200.0

    try:
        # Setup both cases (with setFields for temperature initialization)
        print("\n=== Setting up test cases ===")
        setup_case(
            source_case, test_case_custom, end_time, write_interval, run_setfields=True
        )
        setup_case(
            source_case, test_case_native, end_time, write_interval, run_setfields=True
        )

        # Run custom solver
        print("\n=== Running SimpleSolver ===")
        original_dir = Path.cwd()
        os.chdir(test_case_custom)
        try:
            run(["simpleSolver"])
        finally:
            os.chdir(original_dir)

        # Run native buoyantBoussinesqPimpleFoam
        print("\n=== Running native buoyantBoussinesqPimpleFoam ===")
        result = subprocess.run(
            ["buoyantBoussinesqPimpleFoam", "-case", str(test_case_native)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"buoyantBoussinesqPimpleFoam failed: {result.stderr}"
        )

        # Compare fields using helper
        all_match, failed_fields, failed_details = compare_solver_fields(
            test_case_custom,
            test_case_native,
            FIELDS_TO_COMPARE,
            rtol=1e-10,
            atol=1e-15,
        )

        if not all_match:
            detail_parts = []
            for field in failed_fields:
                max_abs_diff, max_rel_diff = failed_details.get(
                    field, (float("nan"), float("nan"))
                )
                detail_parts.append(
                    f"{field}(max_abs={max_abs_diff:.3e}, max_rel={max_rel_diff:.3e})"
                )
            failure_msg = (
                "Field values differ between solvers. Failed fields with max deviations: "
                + ", ".join(detail_parts)
            )
            assert False, failure_msg

        print("\n=== Test PASSED: Results match exactly ===")

    finally:
        # Clean up test cases
        import shutil

        for test_case in [test_case_custom, test_case_native]:
            if test_case.exists():
                shutil.rmtree(test_case)
                print(f"Cleaned up: {test_case}")
