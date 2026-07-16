# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared damBreak comparison harness: incompressibleVoF vs a native solver.

Both damBreak comparison tests (MULES vs interFoam, isoAdvector vs
interIsoFoam) run the identical sequence — set up two copies of a damBreak
tutorial, run the Python solver in one and the native OpenFOAM solver in the
other, then compare the written fields — so the whole body lives here and each
test is a one-line parameterization.
"""

import os
import shutil
import subprocess
from pathlib import Path

from neofoam.solver.incompressibleVoF import run

# Re-use the comparison helpers from the sibling incompressibleFluid test
# package (``test/solver`` is a package, so a ``..`` relative import reaches it).
from ..incompressibleFluid.comparison_helpers import (
    compare_solver_fields,
    setup_case,
)

# Disable OpenFOAM floating point exception trapping
os.environ["FOAM_SIGFPE"] = ""

# Fields to compare between solvers
FIELDS_TO_COMPARE = [
    ("alpha.water", "volScalarField"),
    ("U", "volVectorField"),
    ("p_rgh", "volScalarField"),
]


def run_dambreak_comparison(
    tutorial_name: str,
    native_solver: str,
    case_prefix: str,
    end_time: float = 0.05,
    write_interval: float = 0.05,
) -> None:
    """Run ``tutorials/<tutorial_name>`` with incompressibleVoF and with the
    native ``native_solver`` binary, then assert the written fields match to
    machine precision (rtol=atol=1e-10).

    ``case_prefix`` names the two scratch case directories under
    ``test_cases/`` (``<case_prefix>_incompressibleVoF`` /
    ``<case_prefix>_<native_solver>``), which are removed afterwards.
    """
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / tutorial_name

    test_case_custom = repo_root / "test_cases" / f"{case_prefix}_incompressibleVoF"
    test_case_native = repo_root / "test_cases" / f"{case_prefix}_{native_solver}"

    try:
        # ------------------------------------------------------------------ #
        # Setup both cases (blockMesh + setFields)                            #
        # ------------------------------------------------------------------ #
        print("\n=== Setting up test cases ===")
        for test_case in (test_case_custom, test_case_native):
            setup_case(
                source_case,
                test_case,
                end_time,
                write_interval,
                run_setfields=True,
            )

        # ------------------------------------------------------------------ #
        # Run incompressibleVoF (advection scheme auto-detected from the      #
        # case's fvSolution)                                                  #
        # ------------------------------------------------------------------ #
        print("\n=== Running incompressibleVoF ===")
        original_dir = Path.cwd()
        os.chdir(test_case_custom)
        try:
            run(["incompressibleVoF"])
        finally:
            os.chdir(original_dir)

        # ------------------------------------------------------------------ #
        # Run the native solver                                               #
        # ------------------------------------------------------------------ #
        print(f"\n=== Running native {native_solver} ===")
        result = subprocess.run(
            [native_solver, "-case", str(test_case_native)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"{native_solver} failed:\n{result.stderr}"

        # ------------------------------------------------------------------ #
        # Compare fields (same schemes, same adaptive-dt sequence)            #
        # ------------------------------------------------------------------ #
        all_match, failed_fields, failed_details = compare_solver_fields(
            test_case_custom,
            test_case_native,
            FIELDS_TO_COMPARE,
            rtol=1e-10,
            atol=1e-10,
        )

        if not all_match:
            detail_parts = []
            for field_name in failed_fields:
                max_abs_diff, max_rel_diff = failed_details.get(
                    field_name, (float("nan"), float("nan"))
                )
                detail_parts.append(
                    f"{field_name}(abs={max_abs_diff:.3e}, rel={max_rel_diff:.3e})"
                )
            assert False, (
                "Field values differ beyond tolerance. Failed fields: "
                + ", ".join(detail_parts)
            )

        print("\n=== Test PASSED: Results match within tolerance ===")

    finally:
        for test_case in (test_case_custom, test_case_native):
            if test_case.exists():
                shutil.rmtree(test_case)
                print(f"Cleaned up: {test_case}")
