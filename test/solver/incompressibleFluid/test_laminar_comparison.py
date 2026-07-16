# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field-by-field comparison for the native ``laminar`` turbulence model.

Exercises the native turbulence path end-to-end: the ``laminar`` spec is built
generically (``spec.create``) and its adapter assembles the momentum stress term
``divDevReff(U)`` from ``nuEff = nu + nut`` (``nut == 0``), drawing ``nu`` from
the native ``Newtonian`` viscosity model. The momentum operation is unchanged —
it still calls ``turbulence.divDevReff(U)``.

The bundled ``tutorials/pitzDaily`` case is switched to ``simulationType
laminar`` for both solvers; incompressibleFluid (PIMPLE, native laminar) must
match native ``pimpleFoam`` (laminar) to round-off on U and p.
"""

import shutil
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
]

# The canonical laminar ``turbulenceProperties`` lives once, as a real case file
# under the turbulence test cases — read it instead of re-encoding the dict here.
_LAMINAR_TURBULENCE_PROPERTIES = (
    Path(__file__).resolve().parents[2]
    / "turbulence"
    / "cases"
    / "laminar"
    / "constant"
    / "turbulenceProperties"
)


def _make_laminar(case_dir: Path) -> None:
    shutil.copyfile(
        _LAMINAR_TURBULENCE_PROPERTIES, case_dir / "constant" / "turbulenceProperties"
    )


def test_laminar_solver_comparison(tmp_path: Path) -> None:
    """Native laminar incompressibleFluid vs native pimpleFoam (laminar)."""
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily"

    end_time = 0.01
    write_interval = 0.01

    # Build custom case: copy, mesh, patch timing.
    custom_case = (
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
    ).build_at(tmp_path / "laminar_custom_solver")
    _make_laminar(custom_case.path)

    # Build native case: copy, mesh, patch timing.
    native_case = (
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
    ).build_at(tmp_path / "laminar_native_solver")
    _make_laminar(native_case.path)

    with cwd(custom_case.path):
        run(["incompressibleFluid"])

    result = subprocess.run(
        ["pimpleFoam", "-case", str(native_case.path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"pimpleFoam failed: {result.stderr}"

    all_match, failed_fields, failed_details = compare_solver_fields(
        custom_case.path,
        native_case.path,
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
