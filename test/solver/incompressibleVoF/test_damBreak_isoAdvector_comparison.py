# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Comparison test for damBreak: incompressibleVoF (isoAdvector) vs native interIsoFoam.

The ``damBreak_isoAdvector`` tutorial selects the geometric advection scheme via
``advectionScheme isoAdvector;`` in ``system/fvSolution`` (with the isoAdvector
controls in the ``"alpha.water.*"`` solver sub-dict). isoAdvector is a pure
function of ``(alpha1, phi, U, deltaT)``; both solvers read identical controls
and share the same adaptive-dt routines (``pybFoam.computeCFLNumber`` and the
Python-composed ``compute_alpha_courant_number``), so they follow the same time-step sequence
deterministically and results must match to machine precision (rtol=1e-10).

Companion to ``test_damBreak_comparison.py`` (the MULES scheme vs interFoam).
"""

import os
import subprocess
from pathlib import Path

from neofoam.solver.incompressibleVoF import run

# Re-use the comparison helpers from the incompressibleFluid test package.
# ``test/solver`` is not a package, so incompressibleFluid / incompressibleVoF
# are sibling top-level packages on sys.path — reach the helper by absolute
# import rather than a ``..`` relative one (which would escape the top level).
from incompressibleFluid.comparison_helpers import (
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


def test_damBreak_isoAdvector_solver_comparison():
    """Compare incompressibleVoF (isoAdvector) against native interIsoFoam."""

    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "damBreak_isoAdvector"

    test_case_custom = repo_root / "test_cases" / "damBreak_iso_incompressibleVoF"
    test_case_native = repo_root / "test_cases" / "damBreak_iso_interIsoFoam"

    end_time = 0.05
    write_interval = 0.05

    try:
        # ------------------------------------------------------------------ #
        # Setup both cases (blockMesh + setFields)                            #
        # ------------------------------------------------------------------ #
        print("\n=== Setting up test cases ===")
        setup_case(
            source_case,
            test_case_custom,
            end_time,
            write_interval,
            run_setfields=True,
        )
        setup_case(
            source_case,
            test_case_native,
            end_time,
            write_interval,
            run_setfields=True,
        )

        # ------------------------------------------------------------------ #
        # Run incompressibleVoF (auto-detects isoAdvector from fvSolution)    #
        # ------------------------------------------------------------------ #
        print("\n=== Running incompressibleVoF (isoAdvector) ===")
        original_dir = Path.cwd()
        os.chdir(test_case_custom)
        try:
            run(["incompressibleVoF"])
        finally:
            os.chdir(original_dir)

        # ------------------------------------------------------------------ #
        # Run native interIsoFoam                                            #
        # ------------------------------------------------------------------ #
        print("\n=== Running native interIsoFoam ===")
        result = subprocess.run(
            ["interIsoFoam", "-case", str(test_case_native)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"interIsoFoam failed:\n{result.stderr}"

        # ------------------------------------------------------------------ #
        # Compare fields (bitwise: same scheme, same dt sequence)             #
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
        import shutil

        for test_case in [test_case_custom, test_case_native]:
            if test_case.exists():
                shutil.rmtree(test_case)
                print(f"Cleaned up: {test_case}")
