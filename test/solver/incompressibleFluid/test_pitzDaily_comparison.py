# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Comparison test for pitzDaily case: SimpleSolver vs native pimpleFoam.
Tests that both solvers produce matching results by loading and comparing
the actual volScalarField/volVectorField data from disk.
"""

import os
import subprocess
from pathlib import Path
from foamadapter.solver.incompressibleFluid import run
import pytest

from .comparison_helpers import (
    requires_openfoam,
    setup_case,
    compare_solver_fields,
)

# Fields to compare between solvers: (field_name, field_type)
FIELDS_TO_COMPARE = [
    ("U", "volVectorField"),
    ("p", "volScalarField"),
    ("k", "volScalarField"),
    ("epsilon", "volScalarField"),
]


@requires_openfoam
def test_pitzDaily_solver_comparison():
    """Compare SimpleSolver against native pimpleFoam on pitzDaily case."""

    # Setup paths
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily"

    test_case_custom = repo_root / "test_cases" / "pitzDaily_custom_solver"
    test_case_native = repo_root / "test_cases" / "pitzDaily_native_solver"

    # Test parameters
    end_time = 0.01
    write_interval = 0.01

    try:
        # Setup both cases
        print("\n=== Setting up test cases ===")
        setup_case(source_case, test_case_custom, end_time, write_interval)
        setup_case(source_case, test_case_native, end_time, write_interval)

        # Run custom solver
        print("\n=== Running SimpleSolver ===")
        original_dir = Path.cwd()
        os.chdir(test_case_custom)
        try:
            run(["simpleSolver"])
        finally:
            os.chdir(original_dir)

        # Run native pimpleFoam
        print("\n=== Running native pimpleFoam ===")
        result = subprocess.run(
            ["pimpleFoam", "-case", str(test_case_native)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"pimpleFoam failed: {result.stderr}"

        # Compare fields using helper
        all_match, failed_fields = compare_solver_fields(
            test_case_custom,
            test_case_native,
            FIELDS_TO_COMPARE,
            rtol=1e-10,
            atol=1e-15,
        )

        if not all_match:
            failure_msg = f"Field values differ between solvers. Failed fields: {', '.join(failed_fields)}"
            assert False, failure_msg

        print("\n=== Test PASSED: Results match exactly ===")

    finally:
        # Clean up test cases
        import shutil

        for test_case in [test_case_custom, test_case_native]:
            if test_case.exists():
                shutil.rmtree(test_case)
                print(f"Cleaned up: {test_case}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
