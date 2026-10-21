# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field comparison for a PIMPLE dict that exercises the whole inner-loop control.

``tutorials/pitzDaily`` already matches native ``pimpleFoam`` to round-off with
its shipped ``PIMPLE { nCorrectors 2; nNonOrthogonalCorrectors 0; }`` — the
combination that hides every inner-loop control deviation. This test re-points
the same case at the settings that expose them, all three at once:

* ``nNonOrthogonalCorrectors 1`` — a non-final non-orthogonal pass must solve on
  the loose ``p`` settings even inside the last pressure corrector
  (``pimpleControl::finalInnerIter()`` is
  ``corrPISO_ == nCorrPISO_ && corrNonOrtho_ == nNonOrthCorr_ + 1``);
* ``nOuterCorrectors 2`` + a ``p`` field relaxation factor — ``pEqn.H`` relaxes
  ``p`` explicitly between pressure correctors;
* ``turbOnFinalIterOnly no`` — ``k``/``epsilon`` are corrected inside *every*
  outer iteration (``pimpleFoam.C``'s ``if (pimple.turbCorr())``), not once
  after the loop.

Tolerance: the same round-off band as the sibling pitzDaily comparison
(``rtol 1e-10``) — with identical solver settings and identical iteration
sequence the two runs execute the same arithmetic, so anything above round-off
is a control deviation.
"""

import subprocess
from pathlib import Path

import pytest

from neofoam.solver.incompressibleFluid import run
from neofoam.tooling.casebuild import block_mesh, from_template, patch

from .._run_case import cwd
from .comparison_helpers import compare_solver_fields

FIELDS_TO_COMPARE = [
    ("U", "volVectorField"),
    ("p", "volScalarField"),
    ("k", "volScalarField"),
    ("epsilon", "volScalarField"),
]

# 10 steps at the tutorial's deltaT: all three deviations show up in the very
# first time step, and a short run keeps the *failing* direction cheap too (a
# wrongly relaxed / wrongly converged pressure loop turns into a slow crawl).
_CONTROL_DICT = {
    "endTime": 0.001,
    "writeControl": "adjustable",
    "writeInterval": 0.001,
}

_FV_SOLUTION = {
    "PIMPLE.nOuterCorrectors": 2,
    "PIMPLE.nCorrectors": 2,
    "PIMPLE.nNonOrthogonalCorrectors": 1,
    "PIMPLE.turbOnFinalIterOnly": False,
    # flat relaxationFactors: OpenFOAM's backwards-compatible reading puts the
    # p entry in the *field* table (explicit p.relax()) and both in the equation
    # table (UEqn.relax()).
    "relaxationFactors.U": 0.8,
    "relaxationFactors.p": 0.8,
}


def test_outer_corrector_loop_matches_native_pimpleFoam(tmp_path: Path) -> None:
    """incompressibleFluid vs pimpleFoam with non-orthogonal + relaxed outer correctors."""
    source_case = Path(__file__).parents[3] / "tutorials" / "pitzDaily"

    def build(name: str) -> Path:
        case = (
            from_template(source_case)
            | block_mesh()
            | patch("system/controlDict", _CONTROL_DICT)
            | patch("system/fvSolution", _FV_SOLUTION)
        ).build_at(tmp_path / name)
        return case.path

    custom_case = build("pimple_correctors_custom")
    native_case = build("pimple_correctors_native")

    with cwd(custom_case):
        run(["incompressibleFluid"])

    result = subprocess.run(
        ["pimpleFoam", "-case", str(native_case)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"pimpleFoam failed: {result.stderr}"

    all_match, failed_fields, failed_details = compare_solver_fields(
        custom_case,
        native_case,
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
