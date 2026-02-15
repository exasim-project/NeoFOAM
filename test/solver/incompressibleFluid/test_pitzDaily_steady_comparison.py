# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Comparison test for steady pitzDaily case: NeoFOAM solver vs native simpleFoam.
Tests that both solvers produce matching results by loading and comparing
actual volScalarField/volVectorField data from disk.
"""

import os
import subprocess
from pathlib import Path

from neofoam.solver.incompressibleFluid import run

from .comparison_helpers import (
    compare_solver_fields,
    requires_openfoam,
    setup_case,
)

# Disable OpenFOAM floating point exception trapping before solver runs.
os.environ["FOAM_SIGFPE"] = ""

# Fields to compare between solvers: (field_name, field_type)
FIELDS_TO_COMPARE = [
    ("U", "volVectorField"),
    ("p", "volScalarField"),
    ("k", "volScalarField"),
    ("epsilon", "volScalarField"),
]


@requires_openfoam
def test_pitzDaily_steady_solver_comparison():
    """Compare NeoFOAM solver against native simpleFoam on pitzDaily_steady case."""

    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily_steady"

    test_case_custom = repo_root / "test_cases" / "pitzDaily_steady_custom_solver"
    test_case_native = repo_root / "test_cases" / "pitzDaily_steady_native_solver"

    # Keep run short for CI, while ensuring at least one written result.
    end_time = 2.0
    write_interval = 1.0

    try:
        print("\n=== Setting up test cases ===")
        setup_case(source_case, test_case_custom, end_time, write_interval)
        setup_case(source_case, test_case_native, end_time, write_interval)

        print("\n=== Running NeoFOAM solver ===")
        original_dir = Path.cwd()
        os.chdir(test_case_custom)
        try:
            run(["simpleSolver"])
        finally:
            os.chdir(original_dir)

        print("\n=== Running native simpleFoam ===")
        result = subprocess.run(
            ["simpleFoam", "-case", str(test_case_native)],
            capture_output=True,
            text=True,
            timeout=240,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, f"simpleFoam failed: {result.stderr}"

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
        import shutil

        for test_case in [test_case_custom, test_case_native]:
            if test_case.exists():
                shutil.rmtree(test_case)
                print(f"Cleaned up: {test_case}")
