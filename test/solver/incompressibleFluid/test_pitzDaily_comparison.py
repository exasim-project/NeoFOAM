# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field-by-field comparison: incompressibleFluid (PIMPLE) vs native pimpleFoam.

Both solvers run on the bundled ``tutorials/pitzDaily`` case for a short
time interval. The volScalar/volVectorField outputs are loaded from disk
and compared numerically; the solvers should match to round-off.
"""

import subprocess
from pathlib import Path

import pytest

from neofoam.tooling.casebuild import from_template, block_mesh, patch
from neofoam.solver.incompressibleFluid import run

from .._run_case import cwd
from .comparison_helpers import (
    compare_solver_fields,
)

FIELDS_TO_COMPARE = [
    ("U", "volVectorField"),
    ("p", "volScalarField"),
    ("k", "volScalarField"),
    ("epsilon", "volScalarField"),
]


def test_pitzDaily_solver_comparison(tmp_path: Path) -> None:
    """Compare incompressibleFluid against native pimpleFoam on pitzDaily."""
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily"

    end_time = 0.01
    write_interval = 0.01

    # Build cases using casebuild pipeline
    test_case_custom = (
        from_template(source_case)
        | block_mesh()
        | patch(
            "system/controlDict",
            {
                "endTime": end_time,
                "writeControl": "adjustable",
                "writeInterval": write_interval,
            },
        )
    ).build_at(tmp_path / "pitzDaily_custom_solver")

    test_case_native = (
        from_template(source_case)
        | block_mesh()
        | patch(
            "system/controlDict",
            {
                "endTime": end_time,
                "writeControl": "adjustable",
                "writeInterval": write_interval,
            },
        )
    ).build_at(tmp_path / "pitzDaily_native_solver")

    # Run custom solver
    with cwd(test_case_custom.path):
        run(["incompressibleFluid"])

    # Run native solver
    result = subprocess.run(
        ["pimpleFoam", "-case", str(test_case_native.path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"pimpleFoam failed: {result.stderr}"

    # Compare fields
    all_match, failed_fields, failed_details = compare_solver_fields(
        test_case_custom.path,
        test_case_native.path,
        FIELDS_TO_COMPARE,
        rtol=1e-10,
        atol=1e-15,
    )

    if not all_match:
        parts = []
        for fname in failed_fields:
            max_abs, max_rel = failed_details.get(fname, (float("nan"), float("nan")))
            parts.append(f"{fname}(abs={max_abs:.3e}, rel={max_rel:.3e})")
        pytest.fail("Field values differ between solvers: " + ", ".join(parts))
